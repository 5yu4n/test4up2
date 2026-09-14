package router

import (
	"bufio"
	"bytes"
	"compress/gzip"
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// compressUpstreamBody reduces wire time for the large conversation/tool
// payloads that dominate interactive requests. TokenRouter accepts gzip
// request bodies; only use it when it materially shrinks the payload so small
// requests never pay compression overhead.
func compressUpstreamBody(body []byte) ([]byte, bool) {
	if len(body) < 64*1024 {
		return body, false
	}
	var compressed bytes.Buffer
	compressed.Grow(len(body) / 2)
	zw, err := gzip.NewWriterLevel(&compressed, gzip.BestSpeed)
	if err != nil {
		return body, false
	}
	if _, err := zw.Write(body); err != nil {
		_ = zw.Close()
		return body, false
	}
	if err := zw.Close(); err != nil || compressed.Len() >= len(body) {
		return body, false
	}
	return compressed.Bytes(), true
}

// prepareOutboundBody applies the optional wire compression used by the
// TokenRouter upstream. Keeping the JSON form separate lets a retry compact a
// large conversation after a provider rate-limit without trying to decode a
// gzip stream.
func prepareOutboundBody(body []byte, allowCompression bool) ([]byte, bool) {
	if !allowCompression {
		return body, false
	}
	return compressUpstreamBody(body)
}

const (
	// A free GLM line can accept the compacted form used by the normal
	// 170k-token conversations. Larger histories are still attempted intact
	// first; this cap is only a recovery path after the provider throttles one.
	largeContextRetryThreshold = 768 * 1024
	// Histories above this size are the ones observed to spend the full 150s
	// header budget before the provider decides whether to serve them. Start
	// those requests with the same bounded suffix we would otherwise use after
	// a timeout. Keeping the threshold high leaves ordinary large requests
	// byte-for-byte intact while avoiding the known multi-minute prefill trap.
	preemptiveContextThreshold = 2 * 1024 * 1024
	compactContextTargetBytes  = 768 * 1024
)

// compactOpenAIContext keeps the system prompt and the newest conversation
// suffix while dropping the oldest tool-heavy turns. It is deliberately used
// only as a retry fallback: normal requests retain the caller's full context.
// The resulting suffix never starts with a tool result, which avoids sending
// an orphaned tool message after the cut point.
func compactOpenAIContext(body []byte, targetBytes int) ([]byte, bool) {
	if targetBytes <= 0 || len(body) <= targetBytes {
		return body, false
	}

	var envelope map[string]interface{}
	if err := json.Unmarshal(body, &envelope); err != nil {
		return body, false
	}
	rawMessages, ok := envelope["messages"].([]interface{})
	if !ok || len(rawMessages) < 4 {
		return body, false
	}

	system := make([]interface{}, 0, 2)
	conversation := make([]interface{}, 0, len(rawMessages))
	for _, raw := range rawMessages {
		message, ok := raw.(map[string]interface{})
		if !ok {
			conversation = append(conversation, raw)
			continue
		}
		role, _ := message["role"].(string)
		if role == "system" {
			system = append(system, raw)
		} else {
			conversation = append(conversation, raw)
		}
	}
	if len(conversation) < 2 {
		return body, false
	}

	// Start at a user or plain assistant boundary whenever possible. This
	// preserves a coherent latest turn and avoids an orphaned tool result.
	start := len(conversation) - 1
	for start > 0 {
		message, _ := conversation[start].(map[string]interface{})
		role, _ := message["role"].(string)
		if role == "tool" {
			start--
			continue
		}
		if role == "user" {
			break
		}
		if role == "assistant" {
			if _, hasCalls := message["tool_calls"]; !hasCalls {
				break
			}
		}
		start--
	}

	marker := map[string]interface{}{
		"role":    "system",
		"content": "Earlier conversation history was compacted after an upstream rate limit; the newest context is preserved.",
	}
	var best []byte
	for idx := start; idx >= 0; idx-- {
		message, _ := conversation[idx].(map[string]interface{})
		role, _ := message["role"].(string)
		if role == "tool" {
			continue
		}
		candidateMessages := make([]interface{}, 0, len(system)+2+len(conversation)-idx)
		candidateMessages = append(candidateMessages, system...)
		candidateMessages = append(candidateMessages, marker)
		candidateMessages = append(candidateMessages, conversation[idx:]...)
		envelope["messages"] = candidateMessages
		candidate, err := json.Marshal(envelope)
		if err != nil {
			return body, false
		}
		if len(candidate) <= targetBytes {
			best = candidate
			continue
		}
		break
	}
	if len(best) > 0 {
		return best, true
	}
	return body, false
}

// upstreamHeaderBudget gives each key a first-byte window. The budget only
// covers response headers, so an SSE stream is not cut off after it has
// started. GLM reasoning and large-context prefill can legitimately take tens
// of seconds; a short 8-12s watchdog turns healthy calls into local 502s.
func upstreamHeaderBudget(key *KeyItem, requestBytes int64) time.Duration {
	budget := 60 * time.Second
	if requestBytes >= 64*1024 {
		budget = 120 * time.Second
	}
	if key != nil && key.LatencyEMAMs > 0 {
		adaptive := time.Duration(key.LatencyEMAMs*2) * time.Millisecond
		if adaptive > budget {
			budget = adaptive
		}
	}
	if budget > 150*time.Second {
		budget = 150 * time.Second
	}
	if requestBytes >= 256*1024 {
		budget = 150 * time.Second
	}
	return budget
}

func upstreamQuery(rawQuery string, anthropicBridge bool) string {
	if rawQuery == "" {
		return ""
	}
	values, err := url.ParseQuery(rawQuery)
	if err != nil {
		return rawQuery
	}
	if anthropicBridge {
		// `beta=true` is an Anthropic compatibility flag. Leaving it on the
		// OpenAI chat endpoint selects a slower compatibility path on some
		// TokenRouter deployments, so remove only that irrelevant flag.
		values.Del("beta")
	}
	return values.Encode()
}

// forwardSSEChunk preserves the exact SSE bytes while flushing each complete
// event separately when a provider coalesces several events in one network
// read. Incomplete fragments are still forwarded immediately; the client SSE
// parser joins them across writes, so this never adds a waiting window.
func forwardSSEChunk(write func([]byte) error, pending *[]byte, chunk []byte) error {
	*pending = append(*pending, chunk...)
	for len(*pending) > 0 {
		lfBoundary := bytes.Index(*pending, []byte("\n\n"))
		crlfBoundary := bytes.Index(*pending, []byte("\r\n\r\n"))
		boundary, boundaryLen := lfBoundary, 2
		if boundary < 0 || (crlfBoundary >= 0 && crlfBoundary < boundary) {
			boundary, boundaryLen = crlfBoundary, 4
		}
		if boundary < 0 {
			fragment := append([]byte(nil), (*pending)...)
			*pending = (*pending)[:0]
			return write(fragment)
		}
		event := append([]byte(nil), (*pending)[:boundary+boundaryLen]...)
		*pending = (*pending)[boundary+boundaryLen:]
		if err := write(event); err != nil {
			return err
		}
	}
	return nil
}

const (
	downstreamStreamWriteTimeout = 15 * time.Second
	upstreamBodyIdleTimeout      = 5 * time.Minute
)

// liveSSEWriter owns every downstream stream write, including keep-alive
// comments sent while a request is queued or waiting for upstream headers.
// Stop joins the heartbeat goroutine so it cannot outlive ServeHTTP.
type liveSSEWriter struct {
	w             http.ResponseWriter
	controller    *http.ResponseController
	ctx           context.Context
	anthropic     bool
	pingInterval  time.Duration
	writeTimeout  time.Duration
	mu            sync.Mutex
	lastModelData time.Time
	writeErr      error
	stopOnce      sync.Once
	stopCh        chan struct{}
	doneCh        chan struct{}
}

func startLiveSSEWriter(w http.ResponseWriter, ctx context.Context, anthropic bool, pingInterval time.Duration) (*liveSSEWriter, error) {
	if pingInterval <= 0 {
		pingInterval = 500 * time.Millisecond
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache, no-transform")
	w.Header().Set("X-Accel-Buffering", "no")
	w.Header().Del("Content-Length")
	w.WriteHeader(http.StatusOK)

	s := &liveSSEWriter{
		w:             w,
		controller:    http.NewResponseController(w),
		ctx:           ctx,
		anthropic:     anthropic,
		pingInterval:  pingInterval,
		writeTimeout:  downstreamStreamWriteTimeout,
		lastModelData: time.Now(),
		stopCh:        make(chan struct{}),
		doneCh:        make(chan struct{}),
	}
	go s.run()
	// A comment is valid before Anthropic message_start and before the first
	// OpenAI data event, so strict parsers ignore this early keep-alive.
	if err := s.write([]byte(": ping\n\n"), false); err != nil {
		s.Stop()
		return s, err
	}
	return s, nil
}

func (s *liveSSEWriter) run() {
	defer close(s.doneCh)
	ticker := time.NewTicker(s.pingInterval)
	defer ticker.Stop()
	for {
		select {
		case <-ticker.C:
			s.mu.Lock()
			shouldPing := s.writeErr == nil && time.Since(s.lastModelData) >= s.pingInterval
			s.mu.Unlock()
			if shouldPing {
				if err := s.write([]byte(": ping\n\n"), false); err != nil {
					return
				}
			}
		case <-s.ctx.Done():
			return
		case <-s.stopCh:
			return
		}
	}
}

func (s *liveSSEWriter) write(chunk []byte, modelData bool) error {
	if err := s.ctx.Err(); err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.writeErr != nil {
		return s.writeErr
	}
	if err := s.ctx.Err(); err != nil {
		return err
	}
	if s.writeTimeout > 0 {
		// Renew this deadline for every token and ping. A wedged downstream
		// connection therefore releases its key instead of blocking forever.
		_ = s.controller.SetWriteDeadline(time.Now().Add(s.writeTimeout))
	}
	_, err := s.w.Write(chunk)
	if err == nil {
		err = s.controller.Flush()
		if errors.Is(err, http.ErrNotSupported) {
			err = nil
		}
	}
	if modelData && err == nil {
		s.lastModelData = time.Now()
	}
	if err != nil {
		s.writeErr = err
	}
	return err
}

func (s *liveSSEWriter) WriteModel(chunk []byte) error {
	return s.write(chunk, true)
}

func (s *liveSSEWriter) WriteError(errorType, message string) error {
	if errorType == "" {
		errorType = "api_error"
	}
	if s.anthropic {
		payload, _ := json.Marshal(map[string]interface{}{
			"type":  "error",
			"error": map[string]string{"type": errorType, "message": message},
		})
		return s.write(append(append([]byte("event: error\ndata: "), payload...), '\n', '\n'), false)
	}
	payload, _ := json.Marshal(map[string]interface{}{
		"error": map[string]string{"type": errorType, "message": message},
	})
	return s.write(append(append([]byte("data: "), payload...), '\n', '\n'), false)
}

func (s *liveSSEWriter) Err() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.writeErr
}

func (s *liveSSEWriter) Stop() error {
	s.stopOnce.Do(func() { close(s.stopCh) })
	<-s.doneCh
	s.mu.Lock()
	defer s.mu.Unlock()
	_ = s.controller.SetWriteDeadline(time.Time{})
	return s.writeErr
}

// bodyIdleMonitor closes an upstream body after caller cancellation or a full
// idle window. It does not impose an absolute lifetime on a healthy stream.
type bodyIdleMonitor struct {
	body     io.Closer
	ctx      context.Context
	idle     time.Duration
	lastRead atomic.Int64
	timedOut atomic.Bool
	stopOnce sync.Once
	stopCh   chan struct{}
	doneCh   chan struct{}
}

func startBodyIdleMonitor(ctx context.Context, body io.Closer, idle time.Duration) *bodyIdleMonitor {
	m := &bodyIdleMonitor{body: body, ctx: ctx, idle: idle, stopCh: make(chan struct{}), doneCh: make(chan struct{})}
	m.Touch()
	go m.run()
	return m
}

func (m *bodyIdleMonitor) run() {
	defer close(m.doneCh)
	interval := m.idle / 10
	if interval < 10*time.Millisecond {
		interval = 10 * time.Millisecond
	}
	if interval > time.Second {
		interval = time.Second
	}
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	for {
		select {
		case <-ticker.C:
			if time.Since(time.Unix(0, m.lastRead.Load())) >= m.idle {
				m.timedOut.Store(true)
				_ = m.body.Close()
				return
			}
		case <-m.ctx.Done():
			_ = m.body.Close()
			return
		case <-m.stopCh:
			return
		}
	}
}

func (m *bodyIdleMonitor) Touch() { m.lastRead.Store(time.Now().UnixNano()) }

func (m *bodyIdleMonitor) Stop() {
	m.stopOnce.Do(func() { close(m.stopCh) })
	<-m.doneCh
}

func (m *bodyIdleMonitor) TimedOut() bool { return m.timedOut.Load() }

func readAllUpstreamBody(ctx context.Context, body io.ReadCloser, idle time.Duration) ([]byte, error) {
	monitor := startBodyIdleMonitor(ctx, body, idle)
	defer monitor.Stop()
	var out bytes.Buffer
	buf := make([]byte, 32*1024)
	for {
		n, err := body.Read(buf)
		if n > 0 {
			monitor.Touch()
			_, _ = out.Write(buf[:n])
		}
		if err != nil {
			if monitor.TimedOut() {
				return out.Bytes(), fmt.Errorf("upstream response body idle for %s", idle)
			}
			if err == io.EOF {
				return out.Bytes(), nil
			}
			return out.Bytes(), err
		}
	}
}

type genericSSETerminalTracker struct {
	pending      []byte
	done         bool
	finishReason bool
	streamErr    error
	usage        parsedTokenUsage
}

func (t *genericSSETerminalTracker) Feed(chunk []byte) {
	t.pending = append(t.pending, chunk...)
	for {
		boundary, boundaryLen := bytes.Index(t.pending, []byte("\n\n")), 2
		if crlf := bytes.Index(t.pending, []byte("\r\n\r\n")); boundary < 0 || (crlf >= 0 && crlf < boundary) {
			boundary, boundaryLen = crlf, 4
		}
		if boundary < 0 {
			return
		}
		t.consumeEvent(t.pending[:boundary])
		t.pending = t.pending[boundary+boundaryLen:]
	}
}

func (t *genericSSETerminalTracker) consumeEvent(event []byte) {
	for _, rawLine := range bytes.Split(bytes.ReplaceAll(event, []byte("\r\n"), []byte("\n")), []byte("\n")) {
		line := bytes.TrimSpace(rawLine)
		if !bytes.HasPrefix(line, []byte("data:")) {
			continue
		}
		data := bytes.TrimSpace(bytes.TrimPrefix(line, []byte("data:")))
		if bytes.Equal(data, []byte("[DONE]")) {
			t.done = true
			continue
		}
		if err := openAIStreamError(data); err != nil {
			t.streamErr = err
			continue
		}
		var chunk openAIChunk
		if json.Unmarshal(data, &chunk) == nil {
			mergeTokenUsage(&t.usage, parseTokenUsageMap(chunk.Usage))
			for _, choice := range chunk.Choices {
				if choice.FinishReason != "" {
					t.finishReason = true
				}
			}
		}
	}
}

func (t *genericSSETerminalTracker) Finish() error {
	if len(bytes.TrimSpace(t.pending)) > 0 {
		t.consumeEvent(t.pending)
		t.pending = nil
	}
	if t.streamErr != nil {
		return t.streamErr
	}
	if !t.done && !t.finishReason {
		return errors.New("upstream stream ended before completion")
	}
	return nil
}

func (t *genericSSETerminalTracker) Usage() parsedTokenUsage {
	if t == nil {
		return parsedTokenUsage{}
	}
	return t.usage
}

var hopByHopHeaders = map[string]bool{
	"connection": true, "keep-alive": true, "proxy-authenticate": true,
	"proxy-authorization": true, "te": true, "trailer": true,
	"transfer-encoding": true, "upgrade": true,
}

func copyResponseHeaders(dst, src http.Header, skipContentLength bool) {
	for key, values := range src {
		lower := strings.ToLower(key)
		if hopByHopHeaders[lower] || (skipContentLength && lower == "content-length") {
			continue
		}
		for _, value := range values {
			dst.Add(key, value)
		}
	}
}

func requestFingerprint(body []byte) string {
	sum := sha256.Sum256(body)
	return fmt.Sprintf("%x", sum[:8])
}

type RequestLogItem struct {
	ID              string    `json:"id"`
	Timestamp       time.Time `json:"timestamp"`
	Method          string    `json:"method"`
	Path            string    `json:"path"`
	StatusCode      int       `json:"status_code"`
	LatencyMs       int64     `json:"latency_ms"`
	KeyID           string    `json:"key_id"`
	KeyName         string    `json:"key_name"`
	Model           string    `json:"model,omitempty"`
	Effort          string    `json:"effort,omitempty"`
	RequestBytes    int64     `json:"request_bytes,omitempty"`
	ReadBodyMs      int64     `json:"read_body_ms,omitempty"`
	QueueWaitMs     int64     `json:"queue_wait_ms,omitempty"`
	HeadersMs       int64     `json:"headers_ms,omitempty"`
	UpstreamProto   string    `json:"upstream_proto,omitempty"`
	Streaming       bool      `json:"streaming"`
	ErrorMsg        string    `json:"error_msg"`
	Fingerprint     string    `json:"request_fingerprint,omitempty"`
	Attempts        int       `json:"attempts,omitempty"`
	WireStatusCode  int       `json:"wire_status_code,omitempty"`
	TerminalState   string    `json:"terminal_state,omitempty"`
	TTFTMs          int64     `json:"ttft_ms,omitempty"`
	Chunks          int       `json:"chunks,omitempty"`
	BytesOut        int64     `json:"bytes_out,omitempty"`
	MaxGapMs        int64     `json:"max_gap_ms,omitempty"`
	InputTokens     int64     `json:"input_tokens,omitempty"`
	OutputTokens    int64     `json:"output_tokens,omitempty"`
	TotalTokens     int64     `json:"total_tokens,omitempty"`
	TokensEstimated bool      `json:"tokens_estimated,omitempty"`
}

type ProxyHandler struct {
	pool    *Pool
	client  *http.Client
	logsMu  sync.RWMutex
	logs    []RequestLogItem
	maxLogs int
	logSink *requestLogSink
	usage   *tokenUsageStore
}

// Keep the total header phase bounded even when one alternate key is tried.
// Stream requests use this short ceiling because their headers should arrive
// immediately; standard JSON requests get a separate, longer budget below.
const maxUpstreamHeaderPhase = 150 * time.Second
const maxStandardHeaderPhase = 150 * time.Second

// maxRequestRetryWindow bounds how long one client request may keep rotating
// across keys. After the window the request ends with a retryable 429 so the
// client reconnects fresh instead of holding one connection indefinitely.
const maxRequestRetryWindow = 8 * time.Minute

func requestedModel(body []byte) string {
	if len(body) == 0 {
		return ""
	}
	var envelope struct {
		Model string `json:"model"`
	}
	if err := json.Unmarshal(body, &envelope); err != nil {
		return ""
	}
	return envelope.Model
}

func requestedEffort(body []byte) string {
	if len(body) == 0 {
		return ""
	}
	var envelope struct {
		ReasoningEffort string `json:"reasoning_effort"`
		OutputConfig    struct {
			Effort string `json:"effort"`
		} `json:"output_config"`
	}
	if err := json.Unmarshal(body, &envelope); err != nil {
		return ""
	}
	if envelope.ReasoningEffort != "" {
		return envelope.ReasoningEffort
	}
	return envelope.OutputConfig.Effort
}

func requestedStream(body []byte) bool {
	if len(body) == 0 {
		return false
	}
	var envelope struct {
		Stream bool `json:"stream"`
	}
	return json.Unmarshal(body, &envelope) == nil && envelope.Stream
}

func isFreebuffFlashModel(model string) bool {
	model = strings.ToLower(strings.TrimSpace(model))
	return strings.Contains(model, "glm-5.3-flash") || strings.Contains(model, "glm-5-3-flash")
}

func isGLM53Model(model string) bool {
	model = strings.ToLower(strings.TrimSpace(model))
	return strings.Contains(model, "glm-5.3") || strings.Contains(model, "glm-5-3")
}

func NewProxyHandler(pool *Pool) *ProxyHandler {
	tr := &http.Transport{
		DialContext: func(ctx context.Context, network, addr string) (net.Conn, error) {
			dialer := &net.Dialer{
				Timeout:   10 * time.Second,
				KeepAlive: 30 * time.Second,
			}
			conn, err := dialer.DialContext(ctx, network, addr)
			if err != nil {
				return nil, err
			}
			if tcpConn, ok := conn.(*net.TCPConn); ok {
				_ = tcpConn.SetNoDelay(true) // Disable Nagle: send every token chunk immediately!
			}
			return conn, nil
		},
		MaxIdleConns:        1000,
		MaxIdleConnsPerHost: 200,
		MaxConnsPerHost:     200,
		IdleConnTimeout:     120 * time.Second,
		TLSHandshakeTimeout: 10 * time.Second,
		WriteBufferSize:     32 * 1024,
		ReadBufferSize:      32 * 1024,
		// A stalled upstream must not hold a user request indefinitely, while
		// still allowing GLM reasoning and large-context prefill to produce its
		// first byte. The per-request watchdog below uses the same ceiling.
		ResponseHeaderTimeout: 180 * time.Second,
		ExpectContinueTimeout: 1 * time.Second,
		DisableCompression:    true,
		ForceAttemptHTTP2:     true,
	}

	loadedLogs, logSink := initializeRequestLogs(pool, 100)
	usagePath := ""
	if pool != nil {
		usagePath = pool.configPath + ".usage.json"
	}
	usageHistory := loadedLogs
	if pool != nil && pool.configPath != "" {
		// The dashboard keeps only the newest 100 rows in memory, while the
		// durable JSONL store retains up to 2,000. Migrate the larger history so
		// a first telemetry startup does not silently omit older usage records.
		if history, err := loadRecentRequestLogs(pool.configPath+".requests.jsonl", requestLogKeepRecords); err == nil && len(history) > len(usageHistory) {
			usageHistory = history
		}
	}
	backfillRequestLogUsage(loadedLogs)
	return &ProxyHandler{
		pool: pool,
		client: &http.Client{
			Transport: tr,
			Timeout:   0, // Streaming requests manage their own timeouts
		},
		logs:    loadedLogs,
		maxLogs: 100,
		logSink: logSink,
		usage:   newTokenUsageStore(usagePath, usageHistory),
	}
}

func (ph *ProxyHandler) GetLogs() []RequestLogItem {
	ph.logsMu.RLock()
	defer ph.logsMu.RUnlock()

	result := make([]RequestLogItem, len(ph.logs))
	copy(result, ph.logs)
	return result
}

func (ph *ProxyHandler) addLog(item RequestLogItem) {
	ph.logsMu.Lock()
	if len(ph.logs) >= ph.maxLogs {
		ph.logs = ph.logs[1:]
	}
	ph.logs = append(ph.logs, item)
	ph.logsMu.Unlock()
	if ph.logSink != nil {
		ph.logSink.enqueue(item)
	}
}

func (ph *ProxyHandler) recordTokenUsage(keyID string, measurement tokenUsageMeasurement) {
	if ph == nil || ph.usage == nil {
		return
	}
	ph.usage.add(keyID, measurement)
}

func (ph *ProxyHandler) GetTokenUsage() TokenUsageStats {
	if ph == nil || ph.usage == nil {
		return TokenUsageStats{ByKey: make(map[string]TokenUsageStats)}
	}
	return ph.usage.snapshot()
}

func (ph *ProxyHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	startTime := time.Now()
	if ph.serveAllowedModelCatalog(w, r) {
		return
	}

	// Read body into buffer to allow pass-through and potential retry on 429
	var bodyBytes []byte
	readBodyStart := time.Now()
	if r.Body != nil {
		var err error
		bodyBytes, err = io.ReadAll(r.Body)
		if err != nil {
			http.Error(w, fmt.Sprintf(`{"error":{"message":"failed to read request body: %v"}}`, err), http.StatusBadRequest)
			return
		}
		_ = r.Body.Close()
	}
	readBodyMs := time.Since(readBodyStart).Milliseconds()

	// Model alias normalization for ccswitch & Claude Code. Request effort and
	// thinking fields remain caller-owned and pass through untouched; the pool
	// default only applies when the caller sent no effort hint at all.
	if len(bodyBytes) > 0 && bytes.Contains(bodyBytes, []byte(`"model"`)) {
		var bodyJSON map[string]interface{}
		if json.Unmarshal(bodyBytes, &bodyJSON) == nil {
			modified := false

			if rawModel, ok := bodyJSON["model"].(string); ok {
				targetModel := ph.pool.ResolveModelAlias(rawModel)
				if targetModel != rawModel {
					bodyJSON["model"] = targetModel
					modified = true
				}
			}

			// Speed floor: apply the configured default effort only to requests
			// that carried no effort field of their own. Measured upstream:
			// an explicit effort hint shortens the reasoning phase before any
			// content token flows (GLM treats a missing hint as open-ended),
			// so this materially raises end-to-end tokens/min without ever
			// overriding a caller's explicit choice.
			if requestedEffort(bodyBytes) == "" {
				if defaultEffort := ph.pool.GetDefaultEffort(); defaultEffort != "" && defaultEffort != "caller" {
					if isGLM53Model(fmt.Sprint(bodyJSON["model"])) {
						bodyJSON["reasoning_effort"] = defaultEffort
						modified = true
					}
				}
			}

			if modified {
				if newBytes, err := json.Marshal(bodyJSON); err == nil {
					bodyBytes = newBytes
				}
			}
		}
	}

	// GLM 5.3 Flash is served by the local Freebuff2API bridge. All other
	// models continue to use the configured TokenRouter upstream.
	requestModel := requestedModel(bodyBytes)
	requestEffort := requestedEffort(bodyBytes)
	streamRequested := requestedStream(bodyBytes)
	if requestModel != "" && !ph.pool.IsModelAllowed(requestModel) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		_, _ = fmt.Fprintf(w, `{"error":{"message":"model %q is not enabled on this router","type":"invalid_request_error"}}`, requestModel)
		return
	}
	if requestModel == "" && requestRequiresModel(r) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"error":{"message":"model is required","type":"invalid_request_error"}}`))
		return
	}
	requestFingerprintValue := requestFingerprint(bodyBytes)
	useFreebuff := ph.pool.IsFreebuffEnabled() && isFreebuffFlashModel(requestModel)
	upstreamStr := strings.TrimSuffix(ph.pool.GetUpstreamURL(), "/")
	if useFreebuff {
		upstreamStr = strings.TrimSuffix(ph.pool.GetFreebuffBaseURL(), "/")
	}
	upstreamURL, err := url.Parse(upstreamStr)
	if err != nil {
		http.Error(w, fmt.Sprintf(`{"error":{"message":"invalid upstream URL: %v"}}`, err), http.StatusBadRequest)
		return
	}
	anthropicBridge := !useFreebuff && tokenRouterAnthropicBridge(upstreamURL.Host, r.URL.Path)
	outboundJSONBytes := bodyBytes
	var toolAliases *toolNameAliases
	if anthropicBridge {
		outboundJSONBytes, toolAliases, err = anthropicRequestToOpenAIWithAliases(bodyBytes)
		if err != nil {
			http.Error(w, fmt.Sprintf(`{"error":{"message":"failed to normalize Anthropic request: %v"}}`, err), http.StatusBadRequest)
			return
		}
	}
	// Compress only outbound TokenRouter payloads. The local Freebuff bridge
	// expects its normal JSON wire format, and callers that already supplied a
	// content encoding must remain byte-for-byte compatible.
	inboundEncoding := strings.TrimSpace(strings.ToLower(r.Header.Get("Content-Encoding")))
	canRewriteOutboundJSON := anthropicBridge && (inboundEncoding == "" || inboundEncoding == "identity")
	allowOutboundCompression := !useFreebuff && (inboundEncoding == "" || inboundEncoding == "identity")
	outboundBodyBytes, compressedOutbound := prepareOutboundBody(outboundJSONBytes, allowOutboundCompression)

	// Construct target URL cleanly
	reqPath := r.URL.Path
	upstreamBasePath := upstreamURL.Path

	if upstreamBasePath != "" && strings.HasPrefix(reqPath, upstreamBasePath) {
		reqPath = strings.TrimPrefix(reqPath, upstreamBasePath)
	}
	if anthropicBridge && strings.Contains(reqPath, "/messages") {
		reqPath = strings.Replace(reqPath, "/messages", "/chat/completions", 1)
	}

	targetPath := upstreamBasePath + reqPath
	if !strings.HasPrefix(targetPath, "/") {
		targetPath = "/" + targetPath
	}

	outboundURL := fmt.Sprintf("%s://%s%s", upstreamURL.Scheme, upstreamURL.Host, targetPath)
	if query := upstreamQuery(r.URL.RawQuery, anthropicBridge); query != "" {
		outboundURL += "?" + query
	}

	isAnthropicClient := strings.Contains(r.URL.Path, "/messages") || r.Header.Get("x-api-key") != "" || r.Header.Get("anthropic-version") != ""
	var liveStream *liveSSEWriter
	wireStatusCode := 0
	if streamRequested {
		liveStream, err = startLiveSSEWriter(w, r.Context(), isAnthropicClient, time.Duration(ph.pool.GetSSEPingIntervalMs())*time.Millisecond)
		if err != nil {
			status := http.StatusBadGateway
			terminal := "downstream_error"
			if r.Context().Err() != nil {
				status = 499
				terminal = "client_cancel"
			}
			ph.addLog(RequestLogItem{
				ID: fmt.Sprintf("req-%d", time.Now().UnixNano()), Timestamp: startTime,
				Method: r.Method, Path: r.URL.Path, StatusCode: status, WireStatusCode: http.StatusOK,
				LatencyMs: time.Since(startTime).Milliseconds(), Model: requestModel, Effort: requestEffort,
				RequestBytes: int64(len(bodyBytes)), ReadBodyMs: readBodyMs, Streaming: true,
				ErrorMsg: err.Error(), Fingerprint: requestFingerprintValue, Attempts: 0, TerminalState: terminal,
			})
			return
		}
		wireStatusCode = http.StatusOK
		defer liveStream.Stop()
	}

	// Diagnostic preview of the outbound payload for per-attempt failure logs.
	outboundPreview := fmt.Sprintf("<%d bytes>", len(outboundBodyBytes))
	if len(outboundBodyBytes) <= 512 {
		outboundPreview = string(outboundBodyBytes)
	}

	excludeKeys := make(map[string]bool)
	// Each failed attempt benches and excludes its key, so allowing roughly one
	// pass over every enabled key bounds the loop naturally. Only one
	// attempt's stream is ever forwarded downstream, so replaying an attempt
	// whose response was lost, empty, or a worker-level 5xx can never
	// duplicate client-visible output; the cost is upstream-side
	// regeneration, acceptable on a free-tier pool to keep agents alive.
	maxAttempts := ph.pool.EnabledKeyCount() + 2
	if maxAttempts > 10 {
		maxAttempts = 10
	}
	if useFreebuff {
		// The bridge owns token rotation; retrying the same local request from
		// TokenRouter only adds latency and can duplicate an expensive run.
		maxAttempts = 1
	}
	largeContextRequest := canRewriteOutboundJSON && len(outboundJSONBytes) > largeContextRetryThreshold
	if largeContextRequest && maxAttempts > 4 {
		// Replaying a multi-megabyte context on every key can spend the whole
		// client turn in upstream prefill. Keep a few alternate-key chances, but
		// do not let a rate-limit storm multiply into ten expensive attempts.
		maxAttempts = 4
	}

	var lastErr error
	var last429RetryAfter string
	var rateLimitAttempts int
	contextCompacted := false
	// Compacting is a recovery action for a large Anthropic conversation. Keep
	// it in one closure so every retryable failure (header timeout, transport
	// failure, 429, or the common 502/503 gateway errors) follows the same
	// path. The original body remains untouched until a failure proves that a
	// full-context replay is not making progress.
	compactLargeContext := func(reason string) bool {
		if !largeContextRequest || contextCompacted {
			return false
		}
		compacted, ok := compactOpenAIContext(outboundJSONBytes, compactContextTargetBytes)
		if !ok {
			return false
		}
		beforeBytes := len(outboundJSONBytes)
		outboundJSONBytes = compacted
		outboundBodyBytes, compressedOutbound = prepareOutboundBody(outboundJSONBytes, allowOutboundCompression)
		outboundPreview = fmt.Sprintf("<%d bytes>", len(outboundBodyBytes))
		contextCompacted = true
		log.Printf("[context] req=%s compacted retry body %d -> %d bytes after %s", requestFingerprintValue, beforeBytes, len(outboundJSONBytes), reason)
		return true
	}
	// The full body is still the default for normal conversations. The
	// measured 2 MB+ histories consistently spent the entire first-byte window
	// in upstream prefill, however, so compact those before the first attempt
	// instead of waiting for a timeout that cannot improve the result.
	if canRewriteOutboundJSON && len(outboundJSONBytes) >= preemptiveContextThreshold {
		compactLargeContext("large-context fast path")
	}
	var keyUsed *KeyItem
	var queueWaitMs int64
	var headersMs int64
	var upstreamProto string
	var attemptsUsed int

	for attempt := 1; attempt <= maxAttempts; attempt++ {
		if !useFreebuff && attempt > 1 && time.Since(startTime) >= maxRequestRetryWindow {
			lastErr = errors.New("upstream retry window exhausted")
			break
		}
		// Freebuff owns its own local session/token pool. Do not consume a
		// TokenRouter key or the account-wide concurrency slots for that path;
		// doing so starves regular TokenRouter traffic whenever Flash is busy.
		var keyItem *KeyItem
		var releaseFunc func(bool)
		if useFreebuff {
			releaseFunc = func(bool) {}
		} else {
			// Acquire available key from pool
			acquireStart := time.Now()
			var err error
			keyItem, releaseFunc, err = ph.pool.AcquireKeySized(r.Context(), excludeKeys, int64(len(bodyBytes)))
			queueWaitMs += time.Since(acquireStart).Milliseconds()
			if err != nil {
				lastErr = fmt.Errorf("queue acquire failed: %w", err)
				break
			}
			keyUsed = keyItem
		}

		logKeyID := ""
		logKeyName := ""
		if useFreebuff {
			logKeyID = "freebuff"
			logKeyName = "Freebuff2API"
		} else if keyItem != nil {
			logKeyID = keyItem.Config.ID
			logKeyName = keyItem.Config.Name
		}
		isReleased := false
		safeRelease := func(success bool) {
			if !isReleased {
				isReleased = true
				releaseFunc(success)
			}
		}
		neutralRelease := func() {
			if isReleased {
				return
			}
			isReleased = true
			if keyItem != nil && !useFreebuff {
				ph.pool.AbortKey(keyItem.Config.ID)
				return
			}
			releaseFunc(true)
		}

		// Create outbound HTTP request
		outReq, err := http.NewRequestWithContext(r.Context(), r.Method, outboundURL, bytes.NewReader(outboundBodyBytes))
		if err != nil {
			safeRelease(false)
			if liveStream != nil {
				_ = liveStream.WriteError("api_error", fmt.Sprintf("failed to create upstream request: %v", err))
			} else {
				http.Error(w, fmt.Sprintf(`{"error":{"message":"failed to create outbound request: %v"}}`, err), http.StatusInternalServerError)
			}
			return
		}

		// Preserve end-to-end headers while letting net/http regenerate hop-by-hop
		// framing headers for the rewritten body. In particular, forwarding an
		// incoming Expect: 100-continue can add a needless one-second pause.
		for k, vv := range r.Header {
			if strings.EqualFold(k, "Host") || strings.EqualFold(k, "Authorization") || strings.EqualFold(k, "x-api-key") || strings.EqualFold(k, "api-key") || strings.EqualFold(k, "Content-Length") || strings.EqualFold(k, "Transfer-Encoding") || strings.EqualFold(k, "Expect") || strings.EqualFold(k, "Connection") || strings.EqualFold(k, "Proxy-Connection") {
				continue
			}
			for _, v := range vv {
				outReq.Header.Add(k, v)
			}
		}

		// Freebuff2API reads its own local auth token. Do not leak TokenRouter
		// keys into that local bridge; regular upstream requests still receive the
		// selected key in all compatible header forms.
		if useFreebuff {
			outReq.Header.Del("Authorization")
			outReq.Header.Del("x-api-key")
			outReq.Header.Del("api-key")
		} else {
			outReq.Header.Set("Authorization", "Bearer "+keyItem.Config.Key)
			if isTokenRouterHost(upstreamURL.Host) {
				// TokenRouter authenticates with Bearer only. Avoid sending
				// Anthropic compatibility headers that can trigger an alternate
				// gateway path or extra request inspection. The bridge has already
				// converted the payload and path to OpenAI format.
				outReq.Header.Del("x-api-key")
				outReq.Header.Del("api-key")
				if anthropicBridge {
					for _, header := range []string{
						"anthropic-version",
						"anthropic-beta",
						"anthropic-dangerous-direct-browser-access",
						"anthropic-model",
					} {
						outReq.Header.Del(header)
					}
				}
			} else {
				outReq.Header.Set("x-api-key", keyItem.Config.Key)
				outReq.Header.Set("api-key", keyItem.Config.Key)
			}
		}
		outReq.Header.Set("Accept-Encoding", "identity")
		// TokenRouter can route stream requests through a buffered JSON path when
		// the client omits an explicit SSE preference. Advertise the response
		// format so headers and the first token arrive without that detour.
		if streamRequested {
			outReq.Header.Set("Accept", "text/event-stream")
		}
		// TokenRouter's gateway gives its generic Go client identity a much
		// slower worker path. Keep an explicitly supplied client identity, but
		// use the proven low-latency curl identity when the caller has no
		// meaningful User-Agent (the default for Go/Python SDKs). This applies to
		// standard JSON responses too; limiting it to SSE was the source of the
		// 502-only-on-/messages regression.
		userAgent := strings.ToLower(strings.TrimSpace(outReq.Header.Get("User-Agent")))
		if userAgent == "" || strings.HasPrefix(userAgent, "go-http-client/") || strings.HasPrefix(userAgent, "python-requests/") {
			outReq.Header.Set("User-Agent", "curl/8.21.0")
		}
		if compressedOutbound {
			outReq.Header.Set("Content-Encoding", "gzip")
		}
		outReq.Host = upstreamURL.Host

		// Send request to upstream. Queue time is deliberately excluded from
		// this watchdog: a request that waited for a local slot must still get a
		// full upstream first-byte window after it is dispatched.
		attemptStart := time.Now()
		// Recompute the budget from the body that this attempt actually sends.
		// A compacted recovery retry must not inherit the original multi-megabyte
		// context's forced 150s header window.
		headerBudget := upstreamHeaderBudget(keyItem, int64(len(outboundJSONBytes)))
		if contextCompacted && headerBudget > 120*time.Second {
			headerBudget = 120 * time.Second
		}
		headerPhaseLimit := maxUpstreamHeaderPhase
		if !streamRequested {
			// Non-streaming APIs commonly emit no headers until the model has
			// finished its full response. Applying the SSE budget here was turning
			// healthy 20s completions into proxy-generated 502s.
			headerPhaseLimit = maxStandardHeaderPhase
		}
		if headerBudget > headerPhaseLimit {
			headerBudget = headerPhaseLimit
		}
		if !useFreebuff {
			remainingRetryWindow := maxRequestRetryWindow - time.Since(startTime)
			if remainingRetryWindow <= 0 {
				neutralRelease()
				lastErr = errors.New("upstream retry window exhausted")
				break
			}
			if headerBudget > remainingRetryWindow {
				headerBudget = remainingRetryWindow
			}
		}
		// Every attempt must start from the caller's live context. Reusing the
		// previous attempt request here would inherit its canceled context and
		// make alternate-key retries fail locally without reaching upstream.
		attemptCtx, cancelAttempt := context.WithCancel(r.Context())
		attemptReq := outReq.WithContext(attemptCtx)
		headerTimeout := make(chan struct{})
		headerTimer := time.AfterFunc(headerBudget, func() {
			close(headerTimeout)
			cancelAttempt()
		})
		attemptsUsed++
		resp, err := ph.client.Do(attemptReq)
		headerTimer.Stop()
		if err != nil {
			timedOut := false
			select {
			case <-headerTimeout:
				timedOut = true
			default:
			}
			// net/http can return context.Canceled just as the watchdog fires,
			// before the timeout channel is observed. The attempt context is ours;
			// the caller context distinguishes this from a client disconnect.
			if !timedOut && attemptCtx.Err() != nil && r.Context().Err() == nil && time.Since(attemptStart)+50*time.Millisecond >= headerBudget {
				timedOut = true
			}
			cancelAttempt()
			if r.Context().Err() != nil {
				// The caller went away (for example, its own request deadline
				// fired). Do not retry or cool down a healthy key for work the
				// caller no longer needs.
				if keyItem != nil {
					ph.pool.AbortKey(keyItem.Config.ID)
					isReleased = true
				} else {
					safeRelease(false)
				}
				lastErr = r.Context().Err()
				break
			}
			safeRelease(false)
			if keyItem != nil {
				if timedOut {
					// Feed an actual header timeout back into the latency model so
					// this key is not immediately selected again after cooldown.
					ph.pool.RecordKeyLatency(keyItem.Config.ID, time.Since(attemptStart).Milliseconds())
					// A whole first-byte window with no headers means the key is
					// effectively not serving; bench it long enough that traffic
					// concentrates on live keys instead of burning a full budget
					// on every rotation. The bench escalates on repeats.
					ph.pool.MarkKeyHeaderTimeout(keyItem.Config.ID)
				} else {
					// Transport failures that happen before the watchdog (DNS,
					// connect, TLS) are usually transient; keep the short cooldown.
					ph.pool.MarkKeyCooldown(keyItem.Config.ID, 10*time.Second)
				}
				excludeKeys[keyItem.Config.ID] = true
			}
			if timedOut {
				lastErr = fmt.Errorf("upstream header phase exceeded %s", headerBudget)
			} else {
				lastErr = err
			}
			// A large context that never produces headers is usually stuck in
			// provider-side prefill. Retry the next key with a compacted suffix
			// before spending the remaining window replaying the same payload.
			compactLargeContext(func() string {
				if timedOut {
					return "upstream header timeout"
				}
				return "upstream transport failure"
			}())
			if keyItem != nil {
				log.Printf("[attempts] req=%s attempt=%d key=%s transport_error=%v timedOut=%v budget=%s body=%s",
					requestFingerprintValue, attempt, keyItem.Config.ID, err, timedOut, headerBudget, outboundPreview)
			}
			// Rotate to another key: the failed attempt is excluded and benched,
			// and only one attempt's stream is ever forwarded downstream, so a
			// replay can never duplicate client-visible output. Clients without
			// their own 5xx retry (observed with ZCode agents) stay alive while
			// the keep-alive comments hold their stream open.
			if attempt >= maxAttempts || time.Since(startTime) >= maxRequestRetryWindow {
				break
			}
			continue
		}
		// Keep the user-facing log latency anchored to the original request, but
		// train key selection on upstream header time only. Including local queue
		// wait in the EWMA can wrongly punish a healthy key after a burst.
		upstreamHeadersMs := time.Since(attemptStart).Milliseconds()
		// HeadersMs is deliberately the current attempt's upstream phase. The
		// overall request duration remains in LatencyMs, while QueueWaitMs records
		// local waiting separately; this keeps the dashboard diagnosis honest when
		// a response body or a retry is the slow part.
		headersMs = upstreamHeadersMs
		upstreamProto = resp.Proto
		isStreamingHeader := strings.Contains(resp.Header.Get("Content-Type"), "text/event-stream")
		if !useFreebuff && !isStreamingHeader && resp.StatusCode >= 200 && resp.StatusCode < 400 && (strings.Contains(r.URL.Path, "/messages") || strings.Contains(r.URL.Path, "/chat/completions")) {
			ph.pool.RecordKeyLatency(keyItem.Config.ID, upstreamHeadersMs)
		}

		// Return rate limits as soon as response headers arrive. Some upstreams
		// leave a 429 body open for tens of seconds; waiting for that body made a
		// single rate limit look like a 244s request and then repeated it seven
		// times. Retry-After is preserved for the caller to apply its own policy.
		if resp.StatusCode == http.StatusTooManyRequests {
			rateLimitAttempts++
			retryAfter := resp.Header.Get("Retry-After")
			log.Printf("[attempts] req=%s attempt=%d key=%s upstream_status=429 retry_after=%q body=%s",
				requestFingerprintValue, attempt, logKeyID, retryAfter, outboundPreview)
			retryDuration := parseRetryAfter(retryAfter)
			if retryDuration <= 0 {
				// TokenRouter sometimes omits Retry-After on account/key throttles.
				// Cool the selected key briefly anyway; otherwise the latency-aware
				// picker immediately chooses the same key and creates a 429 storm.
				retryDuration = 5 * time.Second
				retryAfter = "5"
			}
			// Isolate only the key that was rate-limited. Other keys can
			// continue serving. The pool escalates to account-wide backoff only
			// after distinct keys provide evidence of a shared throttle.
			if keyItem != nil {
				ph.pool.MarkKeyCooldown(keyItem.Config.ID, retryDuration)
				if !useFreebuff {
					_ = ph.pool.RecordUpstreamRateLimit(keyItem.Config.ID, retryDuration)
				}
			}
			_ = resp.Body.Close()
			cancelAttempt()
			safeRelease(false)
			// A single key's throttle must not kill the whole request: bench
			// and exclude this key, then rotate to another, exactly like the
			// 5xx path. The keep-alive comments hold the client stream open
			// meanwhile. Clients treat a surfaced rate_limit_error as fatal
			// (observed with ZCode agents), so only the final attempt emits
			// it; a single-key pool surfaces immediately because the next
			// acquire fast-fails once its only key is excluded.
			if keyItem != nil {
				excludeKeys[keyItem.Config.ID] = true
			}
			compactLargeContext("upstream 429")
			maxRateLimitAttempts := maxAttempts
			if largeContextRequest {
				// After one alternate key has also rate-limited a large context,
				// further replays only burn more prefill capacity and delay the
				// caller. Let the caller retry after the provider's own hint.
				maxRateLimitAttempts = 2
			}
			if !useFreebuff && attempt < maxAttempts && rateLimitAttempts < maxRateLimitAttempts && r.Context().Err() == nil &&
				time.Since(startTime) < maxRequestRetryWindow {
				lastErr = fmt.Errorf("upstream rate limit on key %s", logKeyID)
				last429RetryAfter = retryAfter
				continue
			}
			if liveStream != nil {
				_ = liveStream.WriteError("rate_limit_error", fmt.Sprintf("upstream rate limit; retry after %s", retryAfter))
			} else {
				w.Header().Set("Retry-After", retryAfter)
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusTooManyRequests)
				_, _ = w.Write([]byte(`{"error":{"message":"upstream rate limit; retry later","type":"rate_limit_error"}}`))
				wireStatusCode = http.StatusTooManyRequests
			}
			ph.addLog(RequestLogItem{
				ID:             fmt.Sprintf("req-%d", time.Now().UnixNano()),
				Timestamp:      startTime,
				Method:         r.Method,
				Path:           r.URL.Path,
				StatusCode:     resp.StatusCode,
				LatencyMs:      time.Since(startTime).Milliseconds(),
				KeyID:          logKeyID,
				KeyName:        logKeyName,
				Model:          requestModel,
				Effort:         requestEffort,
				RequestBytes:   int64(len(bodyBytes)),
				ReadBodyMs:     readBodyMs,
				QueueWaitMs:    queueWaitMs,
				HeadersMs:      headersMs,
				UpstreamProto:  upstreamProto,
				Streaming:      streamRequested,
				ErrorMsg:       "upstream rate limit",
				Fingerprint:    requestFingerprintValue,
				Attempts:       attemptsUsed,
				WireStatusCode: wireStatusCode,
				TerminalState:  "upstream_error",
			})
			return
		}

		// A 5xx/401 names one key's worker, not the request. Close the possibly
		// slow error body as soon as headers arrive, bench that key briefly,
		// and rotate to another key while the client stream is held open by
		// keep-alive comments. Only the final attempt surfaces the status.
		isRetryableStatus := resp.StatusCode == 401 || (resp.StatusCode >= 500 && resp.StatusCode <= 599)
		if isRetryableStatus || (liveStream != nil && resp.StatusCode >= 400) {
			statusCode := resp.StatusCode
			log.Printf("[attempts] req=%s attempt=%d key=%s upstream_status=%d body=%s",
				requestFingerprintValue, attempt, logKeyID, statusCode, outboundPreview)
			_ = resp.Body.Close()
			cancelAttempt()
			safeRelease(false)
			if keyItem != nil && isRetryableStatus {
				ph.pool.MarkKeyCooldown(keyItem.Config.ID, 2*time.Second)
				excludeKeys[keyItem.Config.ID] = true
			}
			if isRetryableStatus && statusCode >= 502 {
				compactLargeContext(fmt.Sprintf("upstream status %d", statusCode))
			}
			if isRetryableStatus {
				lastErr = fmt.Errorf("upstream returned status %d", statusCode)
				if !useFreebuff && attempt < maxAttempts &&
					time.Since(startTime) < maxRequestRetryWindow {
					continue
				}
				// Do not emit api_error for an exhausted retryable upstream
				// response. For a streaming request the HTTP status is already
				// committed as 200, so the common failure path below sends an
				// explicit rate_limit_error event that ZCode can retry instead of
				// classifying the turn as a non-retryable business error.
				break
			}
			if liveStream != nil {
				_ = liveStream.WriteError("api_error", fmt.Sprintf("upstream returned status %d", statusCode))
			} else {
				if retryAfter := resp.Header.Get("Retry-After"); retryAfter != "" {
					w.Header().Set("Retry-After", retryAfter)
				}
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(statusCode)
				_, _ = w.Write([]byte(fmt.Sprintf(`{"error":{"message":"upstream returned status %d","type":"upstream_error"}}`, statusCode)))
				wireStatusCode = statusCode
			}
			ph.addLog(RequestLogItem{
				ID:             fmt.Sprintf("req-%d", time.Now().UnixNano()),
				Timestamp:      startTime,
				Method:         r.Method,
				Path:           r.URL.Path,
				StatusCode:     statusCode,
				LatencyMs:      time.Since(startTime).Milliseconds(),
				KeyID:          logKeyID,
				KeyName:        logKeyName,
				Model:          requestModel,
				Effort:         requestEffort,
				RequestBytes:   int64(len(bodyBytes)),
				ReadBodyMs:     readBodyMs,
				QueueWaitMs:    queueWaitMs,
				HeadersMs:      headersMs,
				UpstreamProto:  upstreamProto,
				Streaming:      streamRequested,
				ErrorMsg:       fmt.Sprintf("upstream returned status %d", statusCode),
				Fingerprint:    requestFingerprintValue,
				Attempts:       attemptsUsed,
				WireStatusCode: wireStatusCode,
				TerminalState:  "upstream_error",
			})
			return
		}

		var streamTTFTMs int64
		var streamTTFTUpstreamMs int64
		var streamChunks int
		var streamBytesOut int64
		var streamMaxGapMs int64
		var streamLastChunk time.Time
		var streamErrorMsg string
		var streamUsage parsedTokenUsage
		var streamUpstreamBytes int64
		var anthropicConverter *anthropicStreamConverter
		var responseBodyBytes int64
		var responseUsage parsedTokenUsage
		var requestUsage tokenUsageMeasurement
		terminalState := "complete"

		if isStreamingHeader {
			if liveStream == nil {
				liveStream, err = startLiveSSEWriter(w, r.Context(), isAnthropicClient, time.Duration(ph.pool.GetSSEPingIntervalMs())*time.Millisecond)
				if err != nil {
					streamErrorMsg = err.Error()
					terminalState = "downstream_error"
					neutralRelease()
					_ = resp.Body.Close()
					cancelAttempt()
				} else {
					wireStatusCode = http.StatusOK
					defer liveStream.Stop()
				}
			}
			monitor := startBodyIdleMonitor(attemptCtx, resp.Body, upstreamBodyIdleTimeout)

			// Account for and forward a network chunk as one unit. Flushing the
			// bytes immediately avoids an extra proxy-side wait when the upstream
			// sends a partial SSE line; the client can safely buffer SSE framing.
			writeChunk := func(chunk []byte) error {
				now := time.Now()
				if err := liveStream.WriteModel(chunk); err != nil {
					return err
				}
				streamChunks++
				streamBytesOut += int64(len(chunk))
				if streamLastChunk.IsZero() {
					streamTTFTMs = now.Sub(startTime).Milliseconds()
					streamTTFTUpstreamMs = now.Sub(attemptStart).Milliseconds()
				} else if gap := now.Sub(streamLastChunk).Milliseconds(); gap > streamMaxGapMs {
					streamMaxGapMs = gap
				}
				streamLastChunk = now
				return nil
			}

			if streamErrorMsg == "" && anthropicBridge {
				// TokenRouter's native endpoint is OpenAI SSE. Convert each delta
				// into Anthropic events so Claude clients keep their original wire
				// contract while using the faster, documented upstream path.
				// Hold the converter's initial lifecycle events until the first
				// visible token arrives. Some free-tier workers return a well-formed
				// [DONE] stream with no content at all; forwarding its
				// message_start/message_stop pair would commit an empty success and
				// prevent a safe retry on another key (ZCode reports this as a
				// compaction failure).
				pendingAnthropicChunks := make([][]byte, 0, 4)
				flushPendingAnthropic := func() error {
					for _, pending := range pendingAnthropicChunks {
						if err := writeChunk(pending); err != nil {
							return err
						}
					}
					pendingAnthropicChunks = pendingAnthropicChunks[:0]
					return nil
				}
				bufferUntilAnthropicOutput := func(chunk []byte) error {
					if anthropicConverter == nil || !anthropicConverter.hasVisibleOutput() {
						pendingAnthropicChunks = append(pendingAnthropicChunks, append([]byte(nil), chunk...))
						return nil
					}
					if err := flushPendingAnthropic(); err != nil {
						return err
					}
					return writeChunk(chunk)
				}
				anthropicConverter = newAnthropicStreamConverterWithAliases(bufferUntilAnthropicOutput, requestModel, fmt.Sprintf("msg_proxy_%d", startTime.UnixNano()), toolAliases)
				reader := bufio.NewReaderSize(resp.Body, 16*1024)
				for {
					line, rErr := reader.ReadBytes('\n')
					if len(line) > 0 {
						streamUpstreamBytes += int64(len(line))
						monitor.Touch()
					}
					trimmed := bytes.TrimSpace(line)
					if bytes.HasPrefix(trimmed, []byte("data:")) {
						data := bytes.TrimSpace(bytes.TrimPrefix(trimmed, []byte("data:")))
						if writeErr := anthropicConverter.consumeData(data); writeErr != nil {
							streamErrorMsg = writeErr.Error()
							break
						}
						if anthropicConverter.hasVisibleOutput() {
							if writeErr := flushPendingAnthropic(); writeErr != nil {
								streamErrorMsg = writeErr.Error()
								break
							}
						}
					}
					if rErr != nil {
						if monitor.TimedOut() {
							streamErrorMsg = fmt.Sprintf("upstream response body idle for %s", upstreamBodyIdleTimeout)
						} else if rErr != io.EOF {
							streamErrorMsg = rErr.Error()
						} else if !anthropicConverter.done && anthropicConverter.stopReason == "" {
							streamErrorMsg = "upstream stream ended before completion"
						}
						break
					}
				}
				// `[DONE]` finishes inside consumeData. Some providers omit `[DONE]`
				// but do send finish_reason; that is also a valid completion. Never
				// fabricate message_stop for a truncated stream.
				if !anthropicConverter.done && anthropicConverter.stopReason != "" {
					if finishErr := anthropicConverter.finish(); finishErr != nil && streamErrorMsg == "" {
						streamErrorMsg = finishErr.Error()
					}
				}
				if streamErrorMsg == "" && anthropicConverter.hasVisibleOutput() {
					if writeErr := flushPendingAnthropic(); writeErr != nil {
						streamErrorMsg = writeErr.Error()
					}
				} else if streamErrorMsg == "" && anthropicConverter.done && !anthropicConverter.hasVisibleOutput() {
					// A clean transport-level end is not a useful completion when
					// the translated Anthropic stream contains no output. Keep the
					// downstream SSE uncommitted so the retry path can rotate keys.
					pendingAnthropicChunks = pendingAnthropicChunks[:0]
					streamErrorMsg = "upstream stream completed without visible output"
				}
				streamUsage = anthropicConverter.usage()
			} else if streamErrorMsg == "" {
				// Split coalesced SSE events before writing so a provider's large
				// network read does not make the client render a burst all at once.
				// Incomplete fragments are emitted immediately by forwardSSEChunk.
				pendingSSE := make([]byte, 0, 16*1024)
				terminal := &genericSSETerminalTracker{}
				buf := make([]byte, 16*1024)
				for {
					n, rErr := resp.Body.Read(buf)
					if n > 0 {
						streamUpstreamBytes += int64(n)
						monitor.Touch()
						terminal.Feed(buf[:n])
						if writeErr := forwardSSEChunk(writeChunk, &pendingSSE, buf[:n]); writeErr != nil {
							streamErrorMsg = writeErr.Error()
							break
						}
					}
					if rErr != nil {
						if monitor.TimedOut() {
							streamErrorMsg = fmt.Sprintf("upstream response body idle for %s", upstreamBodyIdleTimeout)
						} else if rErr != io.EOF {
							streamErrorMsg = rErr.Error()
						}
						break
					}
				}
				if len(pendingSSE) > 0 {
					if writeErr := writeChunk(pendingSSE); writeErr != nil && streamErrorMsg == "" {
						streamErrorMsg = writeErr.Error()
					}
				}
				if streamErrorMsg == "" && (strings.Contains(r.URL.Path, "/messages") || strings.Contains(r.URL.Path, "/chat/completions")) {
					if terminalErr := terminal.Finish(); terminalErr != nil {
						streamErrorMsg = terminalErr.Error()
					}
				}
				mergeTokenUsage(&streamUsage, terminal.Usage())
			}

			monitor.Stop()
			_ = resp.Body.Close()
			cancelAttempt()
			// A stream that fails before emitting any visible output can be
			// replayed on an alternate key without duplicating generation: the
			// caller has only received keep-alive comments so far, and the
			// alternative is a proxy 502 that forces the client to replay the
			// same POST anyway. Covers empty upstream streams and truncations
			// that happen before the first event.
			if streamErrorMsg != "" && streamChunks == 0 && streamBytesOut == 0 &&
				!useFreebuff && attempt < maxAttempts && r.Context().Err() == nil &&
				time.Since(startTime) < maxRequestRetryWindow &&
				liveStream != nil && liveStream.Err() == nil {
				if keyItem != nil {
					ph.pool.MarkKeyCooldown(keyItem.Config.ID, 10*time.Second)
					excludeKeys[keyItem.Config.ID] = true
				}
				safeRelease(false)
				continue
			}
			downstreamFailed := liveStream != nil && liveStream.Err() != nil
			if streamErrorMsg != "" {
				terminalState = "upstream_error"
				if strings.Contains(streamErrorMsg, "before completion") {
					terminalState = "incomplete"
				}
				if liveStream != nil && r.Context().Err() == nil && !downstreamFailed {
					errorType := "api_error"
					var upstreamErr *upstreamSSEError
					if errors.As(errors.New(streamErrorMsg), &upstreamErr) && upstreamErr.Type != "" {
						errorType = upstreamErr.Type
					}
					if err := liveStream.WriteError(errorType, streamErrorMsg); err != nil {
						downstreamFailed = true
					}
				}
			}
			if r.Context().Err() != nil {
				terminalState = "client_cancel"
				neutralRelease()
			} else if downstreamFailed {
				terminalState = "downstream_error"
				neutralRelease()
			}
		} else {
			// Buffer a standard response before committing its status. This makes
			// an abrupt EOF or failed Anthropic conversion a real 502 instead of a
			// truncated HTTP 200 with a stale Content-Length.
			responseBody, readErr := readAllUpstreamBody(attemptCtx, resp.Body, upstreamBodyIdleTimeout)
			responseBodyBytes = int64(len(responseBody))
			responseUsage = parseTokenUsage(responseBody)
			_ = resp.Body.Close()
			cancelAttempt()
			if readErr != nil {
				streamErrorMsg = readErr.Error()
				terminalState = "upstream_error"
			} else if liveStream != nil {
				streamErrorMsg = "upstream returned a non-SSE response to a streaming request"
				terminalState = "upstream_error"
			} else if anthropicBridge && resp.StatusCode >= 200 && resp.StatusCode < 400 {
				converted, convertErr := openAIJSONToAnthropicWithAliases(responseBody, requestModel, toolAliases)
				if convertErr != nil {
					streamErrorMsg = "failed to convert upstream response: " + convertErr.Error()
					terminalState = "upstream_error"
				} else {
					responseBody = converted
				}
			}

			if streamErrorMsg != "" {
				if r.Context().Err() != nil {
					terminalState = "client_cancel"
					neutralRelease()
				} else if liveStream != nil {
					if err := liveStream.WriteError("api_error", streamErrorMsg); err != nil {
						terminalState = "downstream_error"
						neutralRelease()
					}
				} else {
					w.Header().Set("Content-Type", "application/json")
					w.WriteHeader(http.StatusBadGateway)
					wireStatusCode = http.StatusBadGateway
					_, _ = w.Write([]byte(fmt.Sprintf(`{"error":{"message":"%s","type":"upstream_error"}}`, strings.ReplaceAll(streamErrorMsg, `"`, `'`))))
				}
			} else {
				copyResponseHeaders(w.Header(), resp.Header, anthropicBridge)
				if anthropicBridge {
					w.Header().Set("Content-Type", "application/json")
				}
				w.WriteHeader(resp.StatusCode)
				wireStatusCode = resp.StatusCode
				if _, writeErr := w.Write(responseBody); writeErr != nil {
					streamErrorMsg = writeErr.Error()
					if r.Context().Err() != nil {
						terminalState = "client_cancel"
					} else {
						terminalState = "downstream_error"
					}
					neutralRelease()
				}
			}
		}

		isSuccess := resp.StatusCode >= 200 && resp.StatusCode < 400 && streamErrorMsg == ""
		if isSuccess && isTokenUsagePath(r.URL.Path) {
			if isStreamingHeader {
				// Estimate a missing provider output count from the upstream wire
				// stream. The client-facing Anthropic bridge adds event framing, so
				// using streamBytesOut there would systematically over-count.
				outputBytes := streamUpstreamBytes
				if outputBytes <= 0 {
					outputBytes = streamBytesOut
				}
				requestUsage = measurementFromPayload(outboundJSONBytes, outputBytes, streamUsage)
			} else {
				requestUsage = measurementFromPayload(outboundJSONBytes, responseBodyBytes, responseUsage)
			}
			ph.recordTokenUsage(logKeyID, requestUsage)
		}
		if !useFreebuff && isStreamingHeader && isSuccess {
			// Header latency alone misses the expensive provider-side thinking
			// phase. Train the picker on time to the first emitted token so a key
			// that accepts headers quickly but stalls output is naturally avoided.
			if streamTTFTUpstreamMs > 0 {
				ph.pool.RecordKeyLatency(keyItem.Config.ID, streamTTFTUpstreamMs)
			}
			ph.pool.RecordKeyStreamQuality(keyItem.Config.ID, streamMaxGapMs)
		}
		if !isReleased {
			safeRelease(isSuccess)
		}

		latency := time.Since(startTime).Milliseconds()
		logStatusCode := resp.StatusCode
		if streamErrorMsg != "" {
			logStatusCode = http.StatusBadGateway
			if r.Context().Err() != nil {
				logStatusCode = 499
			}
		}
		ph.addLog(RequestLogItem{
			ID:              fmt.Sprintf("req-%d", time.Now().UnixNano()),
			Timestamp:       startTime,
			Method:          r.Method,
			Path:            r.URL.Path,
			StatusCode:      logStatusCode,
			LatencyMs:       latency,
			KeyID:           logKeyID,
			KeyName:         logKeyName,
			Model:           requestModel,
			Effort:          requestEffort,
			RequestBytes:    int64(len(bodyBytes)),
			ReadBodyMs:      readBodyMs,
			QueueWaitMs:     queueWaitMs,
			HeadersMs:       headersMs,
			UpstreamProto:   upstreamProto,
			Streaming:       isStreamingHeader,
			TTFTMs:          streamTTFTMs,
			Chunks:          streamChunks,
			BytesOut:        streamBytesOut,
			MaxGapMs:        streamMaxGapMs,
			InputTokens:     requestUsage.InputTokens,
			OutputTokens:    requestUsage.OutputTokens,
			TotalTokens:     requestUsage.InputTokens + requestUsage.OutputTokens,
			TokensEstimated: requestUsage.Estimated,
			ErrorMsg:        streamErrorMsg,
			Fingerprint:     requestFingerprintValue,
			Attempts:        attemptsUsed,
			WireStatusCode:  wireStatusCode,
			TerminalState:   terminalState,
		})

		return
	}

	// All retry attempts failed
	latency := time.Since(startTime).Milliseconds()
	errStr := "request failed after retries"
	if lastErr != nil {
		errStr = lastErr.Error()
	}

	keyID := ""
	keyName := ""
	if keyUsed != nil {
		keyID = keyUsed.Config.ID
		keyName = keyUsed.Config.Name
	}
	if useFreebuff {
		keyID = "freebuff"
		keyName = "Freebuff2API"
	}

	failureStatus := http.StatusBadGateway
	if useFreebuff {
		failureStatus = http.StatusServiceUnavailable
	} else {
		// Every usable key failed. Return the standard retryable status with a
		// Retry-After hint instead of a 5xx: clients without built-in 5xx
		// retry (observed with ZCode agents) back off and recover once the
		// key benches expire, instead of treating the request as dead.
		failureStatus = http.StatusTooManyRequests
	}
	if r.Context().Err() != nil {
		// 499 is the conventional access-log status for a client-closed
		// request. The client has already disconnected, so writing a body is
		// both futile and likely to create a second error in the server log.
		failureStatus = 499
	}
	ph.addLog(RequestLogItem{
		ID:            fmt.Sprintf("req-%d", time.Now().UnixNano()),
		Timestamp:     startTime,
		Method:        r.Method,
		Path:          r.URL.Path,
		StatusCode:    failureStatus,
		LatencyMs:     latency,
		KeyID:         keyID,
		KeyName:       keyName,
		Model:         requestModel,
		Effort:        requestEffort,
		RequestBytes:  int64(len(bodyBytes)),
		ReadBodyMs:    readBodyMs,
		QueueWaitMs:   queueWaitMs,
		HeadersMs:     headersMs,
		UpstreamProto: upstreamProto,
		Streaming:     streamRequested,
		ErrorMsg:      errStr,
		Fingerprint:   requestFingerprintValue,
		Attempts:      attemptsUsed,
	})
	if r.Context().Err() != nil {
		return
	}
	// Reuse the upstream's own Retry-After hint when the exhaustion ended on
	// a rate limit; otherwise a generic short backoff keeps clients from
	// hammering a fully saturated pool.
	retryAfterHint := last429RetryAfter
	if retryAfterHint == "" {
		retryAfterHint = "30"
	}
	if liveStream != nil {
		// The SSE channel is already committed with keep-alive comments; end
		// it with an explicit retryable error event instead of a silent EOF,
		// which agents parse as a dropped connection mid-run.
		_ = liveStream.WriteError("rate_limit_error", fmt.Sprintf("upstream request failed: %s; retry after %ss", errStr, retryAfterHint))
	} else {
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("Retry-After", retryAfterHint)
		w.WriteHeader(failureStatus)
		_, _ = w.Write([]byte(fmt.Sprintf(`{"error":{"message":"upstream request failed: %s","type":"rate_limit_error","retry_after":%q}}`, errStr, retryAfterHint)))
	}
}

func requestRequiresModel(r *http.Request) bool {
	if r == nil || r.Method != http.MethodPost {
		return false
	}
	switch strings.TrimSuffix(r.URL.Path, "/") {
	case "/v1/chat/completions", "/chat/completions", "/v1/messages", "/messages", "/v1/embeddings", "/embeddings":
		return true
	default:
		return false
	}
}

func (ph *ProxyHandler) serveAllowedModelCatalog(w http.ResponseWriter, r *http.Request) bool {
	if r == nil || r.Method != http.MethodGet || ph == nil || ph.pool == nil {
		return false
	}
	path := strings.TrimSuffix(r.URL.Path, "/")
	if path != "/v1/models" && path != "/models" {
		return false
	}
	models := ph.pool.GetAllowedModels()
	if len(models) == 0 {
		return false
	}
	data := make([]map[string]string, 0, len(models))
	for _, model := range models {
		data = append(data, map[string]string{
			"id":       model,
			"object":   "model",
			"owned_by": "configured-allowlist",
		})
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]interface{}{
		"object": "list",
		"data":   data,
	})
	return true
}

func parseRetryAfter(value string) time.Duration {
	value = strings.TrimSpace(value)
	if value == "" {
		return 0
	}
	if seconds, err := strconv.ParseInt(value, 10, 64); err == nil && seconds > 0 {
		return time.Duration(seconds) * time.Second
	}
	if when, err := http.ParseTime(value); err == nil {
		if delay := time.Until(when); delay > 0 {
			return delay
		}
	}
	return 0
}
