package router

import (
	"bytes"
	"compress/gzip"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestCompressUpstreamBodyOnlyForLargePayloads(t *testing.T) {
	small := []byte("small payload")
	if got, compressed := compressUpstreamBody(small); compressed || !bytes.Equal(got, small) {
		t.Fatalf("small payload compression = (%v, %v), want unchanged", got, compressed)
	}

	original := bytes.Repeat([]byte(`{"role":"user","content":"repeated context"}`), 5000)
	compressed, ok := compressUpstreamBody(original)
	if !ok || len(compressed) >= len(original) {
		t.Fatalf("large payload was not reduced: original=%d compressed=%d enabled=%v", len(original), len(compressed), ok)
	}
	reader, err := gzip.NewReader(bytes.NewReader(compressed))
	if err != nil {
		t.Fatalf("gzip reader: %v", err)
	}
	decoded, err := io.ReadAll(reader)
	_ = reader.Close()
	if err != nil {
		t.Fatalf("gzip decode: %v", err)
	}
	if !bytes.Equal(decoded, original) {
		t.Fatal("compressed payload did not round-trip")
	}
}

func TestForwardSSEChunkSplitsCoalescedEventsWithoutChangingBytes(t *testing.T) {
	input := []byte("data: one\n\ndata: two\n\n")
	var writes [][]byte
	pending := make([]byte, 0)
	if err := forwardSSEChunk(func(chunk []byte) error {
		writes = append(writes, append([]byte(nil), chunk...))
		return nil
	}, &pending, input); err != nil {
		t.Fatalf("forwardSSEChunk: %v", err)
	}
	if len(writes) != 2 || string(writes[0]) != "data: one\n\n" || string(writes[1]) != "data: two\n\n" {
		t.Fatalf("writes = %q, want two complete SSE events", writes)
	}
	if len(pending) != 0 {
		t.Fatalf("pending = %q, want empty", pending)
	}
}

func TestUpstreamHeaderBudgetUsesObservedLatencyAndBounds(t *testing.T) {
	if got := upstreamHeaderBudget(nil, 0); got != 60*time.Second {
		t.Fatalf("unknown key budget = %s, want 60s", got)
	}
	fast := &KeyItem{LatencyEMAMs: 2500}
	if got := upstreamHeaderBudget(fast, 0); got != 60*time.Second {
		t.Fatalf("fast key budget = %s, want 60s floor", got)
	}
	slow := &KeyItem{LatencyEMAMs: 20000}
	if got := upstreamHeaderBudget(slow, 0); got != 60*time.Second {
		t.Fatalf("slow key budget = %s, want 60s floor", got)
	}
	large := &KeyItem{LatencyEMAMs: 2500}
	if got := upstreamHeaderBudget(large, 2*1024*1024); got != 150*time.Second {
		t.Fatalf("large request budget = %s, want 150s first-byte window", got)
	}
}

func TestUpstreamQueryDropsAnthropicBetaFlagForBridge(t *testing.T) {
	if got := upstreamQuery("beta=true&foo=bar", true); got != "foo=bar" {
		t.Fatalf("bridge query = %q, want foo=bar", got)
	}
	if got := upstreamQuery("beta=true&foo=bar", false); got != "beta=true&foo=bar" {
		t.Fatalf("passthrough query = %q, want beta=true&foo=bar", got)
	}
}

func TestTokenRouterHostDetection(t *testing.T) {
	if !isTokenRouterHost("api.tokenrouter.com") || !isTokenRouterHost("API.TOKENROUTER.COM:443") {
		t.Fatal("TokenRouter host detection failed")
	}
	if isTokenRouterHost("api.example.com") {
		t.Fatal("non-TokenRouter host was detected as TokenRouter")
	}
}

func testPool(t *testing.T, upstream, freebuff string, enabled bool) *Pool {
	t.Helper()
	config := `{"port":8080,"upstream_url":"` + upstream + `","freebuff_base_url":"` + freebuff + `","freebuff_enabled":` + map[bool]string{true: "true", false: "false"}[enabled] + `,"max_queue_seconds":1,"default_effort":"max","keys":[{"id":"k1","name":"K1","key":"test-key","rpm_limit":1,"enabled":true}]}`
	path := filepath.Join(t.TempDir(), "config.json")
	if err := os.WriteFile(path, []byte(config), 0600); err != nil {
		t.Fatal(err)
	}
	p, err := NewPool(path)
	if err != nil {
		t.Fatal(err)
	}
	return p
}

func TestProxyReturns429WithoutSerialRetryDelay(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Retry-After", "3")
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusTooManyRequests)
		time.Sleep(150 * time.Millisecond)
		_, _ = w.Write([]byte(`{"error":"rate limited"}`))
	}))
	defer upstream.Close()

	ph := NewProxyHandler(testPool(t, upstream.URL, "", false))
	ph.pool.maxConcurrentRequests = 2
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(`{"model":"glm-5.3-free"}`))
	rec := httptest.NewRecorder()
	started := time.Now()
	ph.ServeHTTP(rec, req)

	if rec.Code != http.StatusTooManyRequests {
		t.Fatalf("status = %d, want 429", rec.Code)
	}
	if got := rec.Header().Get("Retry-After"); got != "3" {
		t.Fatalf("Retry-After = %q, want 3", got)
	}
	if !ph.pool.keys["k1"].CooldownUntil.After(time.Now().Add(2 * time.Second)) {
		t.Fatal("rate-limited key was not cooled down from Retry-After")
	}
	// A single key's 429 is not account-wide evidence; concurrency may only be
	// lowered after two distinct keys report throttles in the signal window
	// (see TestPoolRequiresDistinctRateLimitedKeysBeforeAIMD).
	if got := ph.pool.GetStats().EffectiveMaxConcurrent; got != 2 {
		t.Fatalf("effective concurrency after single-key throttle = %d, want 2", got)
	}
	if elapsed := time.Since(started); elapsed > 750*time.Millisecond {
		t.Fatalf("429 took %s; likely serial retry/drain regression", elapsed)
	}
}

func TestProxyCoolsKeyWhen429OmitsRetryAfter(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = io.WriteString(w, `{"error":"rate limited"}`)
	}))
	defer upstream.Close()

	ph := NewProxyHandler(testPool(t, upstream.URL, "", false))
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(`{"model":"glm-5.3-free"}`))
	rec := httptest.NewRecorder()
	ph.ServeHTTP(rec, req)

	if rec.Code != http.StatusTooManyRequests {
		t.Fatalf("status = %d, want 429", rec.Code)
	}
	if got := rec.Header().Get("Retry-After"); got != "5" {
		t.Fatalf("fallback Retry-After = %q, want 5", got)
	}
	if remaining := time.Until(ph.pool.keys["k1"].CooldownUntil); remaining < 4*time.Second {
		t.Fatalf("missing Retry-After did not apply fallback cooldown: %s", remaining)
	}
}

func TestParseRetryAfterSupportsSecondsAndHTTPDate(t *testing.T) {
	if got := parseRetryAfter("3"); got != 3*time.Second {
		t.Fatalf("seconds retry-after = %s, want 3s", got)
	}
	when := time.Now().Add(2 * time.Second)
	got := parseRetryAfter(when.UTC().Format(http.TimeFormat))
	if got < 500*time.Millisecond || got > 3*time.Second {
		t.Fatalf("date retry-after = %s, want approximately 2s", got)
	}
	if got := parseRetryAfter("nonsense"); got != 0 {
		t.Fatalf("invalid retry-after = %s, want zero", got)
	}
}

func TestProxyDoesNotRetry403PermissionFailure(t *testing.T) {
	var calls int
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusForbidden)
		_, _ = io.WriteString(w, `{"error":"insufficient quota"}`)
	}))
	defer upstream.Close()

	ph := NewProxyHandler(testPool(t, upstream.URL, "", false))
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"z-ai/glm-5.3-flash","messages":[]}`))
	rec := httptest.NewRecorder()
	ph.ServeHTTP(rec, req)

	if rec.Code != http.StatusForbidden {
		t.Fatalf("status = %d, want 403", rec.Code)
	}
	if calls != 1 {
		t.Fatalf("upstream calls = %d, want one permission check", calls)
	}
}

func TestProxyDoesNotDrainFinalSlow5xxBody(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusBadGateway)
		if flusher, ok := w.(http.Flusher); ok {
			flusher.Flush()
		}
		time.Sleep(3 * time.Second)
		_, _ = io.WriteString(w, "data: upstream error\n\n")
	}))
	defer upstream.Close()

	ph := NewProxyHandler(testPool(t, upstream.URL, "", false))
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(`{"model":"glm-5.3-free","stream":true}`))
	rec := httptest.NewRecorder()
	started := time.Now()
	ph.ServeHTTP(rec, req)

	// The 502 is absorbed internally: the key is benched and, with no other
	// key available, the request ends with a retryable SSE error after the
	// queue deadline — never waiting for the slow 5xx body.
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want committed SSE 200", rec.Code)
	}
	body := rec.Body.String()
	if !strings.Contains(body, "rate_limit_error") || !strings.Contains(body, "queue timeout") {
		t.Fatalf("exhausted-retry outcome missing: %q", body)
	}
	if elapsed := time.Since(started); elapsed >= 2500*time.Millisecond {
		t.Fatalf("final 5xx waited %s; likely drained the slow error body", elapsed)
	}
}

func TestProxyRotatesToAlternateKeyOnUpstream5xx(t *testing.T) {
	var calls int
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		if calls == 1 {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusBadGateway)
			_, _ = io.WriteString(w, `{"error":"worker failed"}`)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"ok":true}`)
	}))
	defer upstream.Close()

	ph := NewProxyHandler(testPool(t, upstream.URL, "", false))
	ph.pool.keys["k2"] = &KeyItem{Config: KeyConfig{ID: "k2", Name: "K2", Key: "test-key-2", RPMLimit: 10, Enabled: true}, Timestamps: make([]time.Time, 0)}
	ph.pool.keyOrder = append(ph.pool.keyOrder, "k2")
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"z-ai/glm-5.3-free","messages":[]}`))
	rec := httptest.NewRecorder()
	ph.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK || rec.Body.String() != `{"ok":true}` {
		t.Fatalf("response = %d %q, want the 5xx absorbed and 200 delivered", rec.Code, rec.Body.String())
	}
	if calls != 2 {
		t.Fatalf("upstream calls = %d, want one rotation retry", calls)
	}
	logs := ph.GetLogs()
	if len(logs) != 1 || logs[0].Attempts != 2 || logs[0].StatusCode != http.StatusOK {
		t.Fatalf("retry log = %#v, want one two-attempt success record", logs)
	}
}

func TestProxyRotatesToAlternateKeyOnUpstream429(t *testing.T) {
	var calls int
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		if calls == 1 {
			w.Header().Set("Content-Type", "application/json")
			w.Header().Set("Retry-After", "3")
			w.WriteHeader(http.StatusTooManyRequests)
			_, _ = io.WriteString(w, `{"error":"rate limited"}`)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"ok":true}`)
	}))
	defer upstream.Close()

	ph := NewProxyHandler(testPool(t, upstream.URL, "", false))
	ph.pool.keys["k2"] = &KeyItem{Config: KeyConfig{ID: "k2", Name: "K2", Key: "test-key-2", RPMLimit: 10, Enabled: true}, Timestamps: make([]time.Time, 0)}
	ph.pool.keyOrder = append(ph.pool.keyOrder, "k2")
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"z-ai/glm-5.3-free","messages":[]}`))
	rec := httptest.NewRecorder()
	ph.ServeHTTP(rec, req)

	// One key's throttle must be absorbed by rotation, not surfaced to a
	// client that treats rate_limit_error as a fatal business error.
	if rec.Code != http.StatusOK || rec.Body.String() != `{"ok":true}` {
		t.Fatalf("response = %d %q, want the 429 absorbed and 200 delivered", rec.Code, rec.Body.String())
	}
	if calls != 2 {
		t.Fatalf("upstream calls = %d, want one rotation retry", calls)
	}
	logs := ph.GetLogs()
	if len(logs) != 1 || logs[0].Attempts != 2 || logs[0].StatusCode != http.StatusOK {
		t.Fatalf("retry log = %#v, want one two-attempt success record", logs)
	}
}

func TestCompactOpenAIContextKeepsSystemAndNewestTurn(t *testing.T) {
	messages := []interface{}{
		map[string]interface{}{"role": "system", "content": "system prompt"},
	}
	for i := 0; i < 24; i++ {
		messages = append(messages,
			map[string]interface{}{"role": "user", "content": strings.Repeat("old context ", 180)},
			map[string]interface{}{"role": "assistant", "content": strings.Repeat("old answer ", 180)},
		)
	}
	messages = append(messages,
		map[string]interface{}{"role": "user", "content": "keep this newest request"},
		map[string]interface{}{"role": "assistant", "content": "keep this newest answer"},
	)
	original, err := json.Marshal(map[string]interface{}{
		"model":    "z-ai/glm-5.3-free",
		"messages": messages,
	})
	if err != nil {
		t.Fatalf("marshal original: %v", err)
	}
	compacted, ok := compactOpenAIContext(original, 32*1024)
	if !ok {
		t.Fatal("large context was not compacted")
	}
	if len(compacted) > 32*1024 || len(compacted) >= len(original) {
		t.Fatalf("compacted size = %d, original = %d", len(compacted), len(original))
	}

	var decoded map[string]interface{}
	if err := json.Unmarshal(compacted, &decoded); err != nil {
		t.Fatalf("unmarshal compacted: %v", err)
	}
	kept, _ := decoded["messages"].([]interface{})
	if len(kept) < 3 {
		t.Fatalf("compacted messages = %d, want system, marker, and newest turn", len(kept))
	}
	if kept[0].(map[string]interface{})["content"] != "system prompt" {
		t.Fatalf("system prompt was not preserved: %#v", kept[0])
	}
	if !strings.Contains(kept[1].(map[string]interface{})["content"].(string), "compacted") {
		t.Fatalf("compaction marker missing: %#v", kept[1])
	}
	last := kept[len(kept)-2].(map[string]interface{})
	if last["content"] != "keep this newest request" {
		t.Fatalf("newest request was not preserved: %#v", last)
	}
	if role, _ := kept[2].(map[string]interface{})["role"].(string); role == "tool" {
		t.Fatalf("compacted suffix starts with orphaned tool message: %#v", kept[2])
	}
}

func TestProxyCompactsLargeAnthropicContextAfter429(t *testing.T) {
	var calls int
	var wireSizes []int
	ph := NewProxyHandler(testPool(t, "https://api.tokenrouter.com/v1", "", false))
	ph.pool.keys["k2"] = &KeyItem{Config: KeyConfig{ID: "k2", Name: "K2", Key: "test-key-2", RPMLimit: 10, Enabled: true}, Timestamps: make([]time.Time, 0)}
	ph.pool.keyOrder = append(ph.pool.keyOrder, "k2")
	ph.client.Transport = roundTripFunc(func(req *http.Request) (*http.Response, error) {
		calls++
		wire, _ := io.ReadAll(req.Body)
		wireSizes = append(wireSizes, len(wire))
		if calls == 1 {
			return &http.Response{
				StatusCode: http.StatusTooManyRequests,
				Header:     http.Header{"Content-Type": []string{"application/json"}},
				Body:       io.NopCloser(strings.NewReader(`{"error":"rate limited"}`)),
				Request:    req,
			}, nil
		}
		return &http.Response{
			StatusCode: http.StatusOK,
			Header:     http.Header{"Content-Type": []string{"application/json"}},
			Body:       io.NopCloser(strings.NewReader(`{"id":"chat_1","model":"z-ai/glm-5.3-free","choices":[{"message":{"content":"ok"},"finish_reason":"stop"}],"usage":{}}`)),
			Request:    req,
		}, nil
	})

	messages := make([]interface{}, 0, 181)
	for i := 0; i < 90; i++ {
		messages = append(messages,
			map[string]interface{}{"role": "user", "content": strings.Repeat("old conversation ", 320)},
			map[string]interface{}{"role": "assistant", "content": strings.Repeat("old response ", 320)},
		)
	}
	messages = append(messages, map[string]interface{}{"role": "user", "content": "new request"})
	body, err := json.Marshal(map[string]interface{}{
		"model":      "z-ai/glm-5.3-free",
		"max_tokens": 8,
		"stream":     false,
		"messages":   messages,
	})
	if err != nil {
		t.Fatalf("marshal request: %v", err)
	}
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", bytes.NewReader(body))
	rec := httptest.NewRecorder()
	ph.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK || calls != 2 {
		t.Fatalf("status=%d calls=%d, want one rate-limit retry and 200", rec.Code, calls)
	}
	if len(wireSizes) != 2 || wireSizes[1] >= wireSizes[0] {
		t.Fatalf("wire sizes = %v, want compacted second attempt", wireSizes)
	}
	logs := ph.GetLogs()
	if len(logs) != 1 || logs[0].StatusCode != http.StatusOK || logs[0].Attempts != 2 {
		t.Fatalf("retry log = %#v, want one successful two-attempt record", logs)
	}
}

func TestProxyCompactsLargeAnthropicContextAfterTransportFailure(t *testing.T) {
	var calls int
	var wireSizes []int
	ph := NewProxyHandler(testPool(t, "https://api.tokenrouter.com/v1", "", false))
	ph.pool.keys["k2"] = &KeyItem{Config: KeyConfig{ID: "k2", Name: "K2", Key: "test-key-2", RPMLimit: 10, Enabled: true}, Timestamps: make([]time.Time, 0)}
	ph.pool.keyOrder = append(ph.pool.keyOrder, "k2")
	ph.client.Transport = roundTripFunc(func(req *http.Request) (*http.Response, error) {
		calls++
		wire, _ := io.ReadAll(req.Body)
		wireSizes = append(wireSizes, len(wire))
		if calls == 1 {
			return nil, context.DeadlineExceeded
		}
		return &http.Response{
			StatusCode: http.StatusOK,
			Header:     http.Header{"Content-Type": []string{"application/json"}},
			Body:       io.NopCloser(strings.NewReader(`{"id":"chat_1","model":"z-ai/glm-5.3-free","choices":[{"message":{"content":"ok"},"finish_reason":"stop"}],"usage":{}}`)),
			Request:    req,
		}, nil
	})

	messages := make([]interface{}, 0, 181)
	for i := 0; i < 90; i++ {
		messages = append(messages,
			map[string]interface{}{"role": "user", "content": strings.Repeat("old conversation ", 320)},
			map[string]interface{}{"role": "assistant", "content": strings.Repeat("old response ", 320)},
		)
	}
	messages = append(messages, map[string]interface{}{"role": "user", "content": "new request"})
	body, err := json.Marshal(map[string]interface{}{
		"model":      "z-ai/glm-5.3-free",
		"max_tokens": 8,
		"stream":     false,
		"messages":   messages,
	})
	if err != nil {
		t.Fatalf("marshal request: %v", err)
	}
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", bytes.NewReader(body))
	rec := httptest.NewRecorder()
	ph.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK || calls != 2 {
		t.Fatalf("status=%d calls=%d, want one transport retry and 200", rec.Code, calls)
	}
	if len(wireSizes) != 2 || wireSizes[1] >= wireSizes[0] {
		t.Fatalf("wire sizes = %v, want compacted second attempt", wireSizes)
	}
	logs := ph.GetLogs()
	if len(logs) != 1 || logs[0].StatusCode != http.StatusOK || logs[0].Attempts != 2 {
		t.Fatalf("retry log = %#v, want one successful two-attempt record", logs)
	}
}

func TestProxyPrecompactsVeryLargeAnthropicContext(t *testing.T) {
	var calls int
	var wireSize int
	ph := NewProxyHandler(testPool(t, "https://api.tokenrouter.com/v1", "", false))
	ph.client.Transport = roundTripFunc(func(req *http.Request) (*http.Response, error) {
		calls++
		wire, _ := io.ReadAll(req.Body)
		wireSize = len(wire)
		return &http.Response{
			StatusCode: http.StatusOK,
			Header:     http.Header{"Content-Type": []string{"application/json"}},
			Body:       io.NopCloser(strings.NewReader(`{"id":"chat_1","model":"z-ai/glm-5.3-free","choices":[{"message":{"content":"ok"},"finish_reason":"stop"}],"usage":{}}`)),
			Request:    req,
		}, nil
	})

	messages := make([]interface{}, 0, 241)
	for i := 0; i < 120; i++ {
		messages = append(messages,
			map[string]interface{}{"role": "user", "content": strings.Repeat("old conversation ", 700)},
			map[string]interface{}{"role": "assistant", "content": strings.Repeat("old response ", 700)},
		)
	}
	messages = append(messages, map[string]interface{}{"role": "user", "content": "new request"})
	body, err := json.Marshal(map[string]interface{}{
		"model":      "z-ai/glm-5.3-free",
		"max_tokens": 8,
		"stream":     false,
		"messages":   messages,
	})
	if err != nil {
		t.Fatalf("marshal request: %v", err)
	}
	if len(body) < preemptiveContextThreshold {
		t.Fatalf("test body = %d bytes, want at least %d", len(body), preemptiveContextThreshold)
	}
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", bytes.NewReader(body))
	rec := httptest.NewRecorder()
	ph.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK || calls != 1 {
		t.Fatalf("status=%d calls=%d, want one precompacted request and 200", rec.Code, calls)
	}
	if wireSize >= len(body) {
		t.Fatalf("wire size = %d, original = %d, want precompacted wire body", wireSize, len(body))
	}
	logs := ph.GetLogs()
	if len(logs) != 1 || logs[0].StatusCode != http.StatusOK || logs[0].Attempts != 1 {
		t.Fatalf("log = %#v, want one successful attempt", logs)
	}
}

func TestProxyExhaustedKeysReturn429WithRetryAfter(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = io.WriteString(w, `{"error":"broken"}`)
	}))
	defer upstream.Close()

	ph := NewProxyHandler(testPool(t, upstream.URL, "", false))
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"z-ai/glm-5.3-free","messages":[]}`))
	rec := httptest.NewRecorder()
	ph.ServeHTTP(rec, req)

	// Every key failed; the client must see the standard retryable status
	// with a Retry-After hint instead of a fatal 5xx.
	if rec.Code != http.StatusTooManyRequests {
		t.Fatalf("status = %d, want retryable 429 after exhausted keys", rec.Code)
	}
	if got := rec.Header().Get("Retry-After"); got != "30" {
		t.Fatalf("Retry-After = %q, want 30", got)
	}
	if !strings.Contains(rec.Body.String(), "rate_limit_error") {
		t.Fatalf("error body = %q, want rate_limit_error type", rec.Body.String())
	}
	logs := ph.GetLogs()
	if len(logs) != 1 || logs[0].StatusCode != http.StatusTooManyRequests {
		t.Fatalf("failure log = %#v, want a single 429 record", logs)
	}
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(req *http.Request) (*http.Response, error) {
	return f(req)
}

func TestProxyExhaustedStreaming5xxUsesRetryableError(t *testing.T) {
	ph := NewProxyHandler(testPool(t, "https://api.tokenrouter.com/v1", "", false))
	for i := 2; i <= 4; i++ {
		id := fmt.Sprintf("k%d", i)
		ph.pool.keys[id] = &KeyItem{Config: KeyConfig{ID: id, Name: id, Key: "test-key-" + id, RPMLimit: 10, Enabled: true}, Timestamps: make([]time.Time, 0)}
		ph.pool.keyOrder = append(ph.pool.keyOrder, id)
	}

	var calls int
	ph.client.Transport = roundTripFunc(func(req *http.Request) (*http.Response, error) {
		calls++
		return &http.Response{
			StatusCode: http.StatusServiceUnavailable,
			Status:     "503 Service Unavailable",
			Header:     make(http.Header),
			Body:       io.NopCloser(strings.NewReader(`{"error":"worker unavailable"}`)),
			Request:    req,
		}, nil
	})
	body := `{"model":"claude-3-5-sonnet","stream":true,"messages":[{"role":"user","content":"` + strings.Repeat("context ", 110000) + `"}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(body))
	rec := httptest.NewRecorder()
	ph.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want committed SSE 200", rec.Code)
	}
	if calls != 4 {
		t.Fatalf("upstream calls = %d, want one attempt per key", calls)
	}
	response := rec.Body.String()
	if !strings.Contains(response, "rate_limit_error") || !strings.Contains(response, "upstream returned status 503") {
		t.Fatalf("response = %q, want explicit retryable 503 error", response)
	}
	if strings.Contains(response, `"type":"api_error"`) {
		t.Fatalf("response = %q, must not classify exhausted 5xx as api_error", response)
	}
}

func TestProxyAnthropicBridgeNormalizesUpstreamRequest(t *testing.T) {
	ph := NewProxyHandler(testPool(t, "https://api.tokenrouter.com/v1", "", false))
	ph.pool.modelMappings["claude-3-5-sonnet"] = "z-ai/glm-5.3-free"
	ph.pool.defaultEffort = "low"
	var captured *http.Request
	var capturedBody []byte
	ph.client.Transport = roundTripFunc(func(req *http.Request) (*http.Response, error) {
		captured = req
		capturedBody, _ = io.ReadAll(req.Body)
		return &http.Response{
			StatusCode: http.StatusOK,
			Header:     http.Header{"Content-Type": []string{"text/event-stream"}},
			Body:       io.NopCloser(strings.NewReader("data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\ndata: [DONE]\n\n")),
			Request:    req,
		}, nil
	})

	req := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(`{"model":"claude-3-5-sonnet","max_tokens":4,"stream":true,"messages":[{"role":"user","content":"Reply OK."}]}`))
	req.Header.Set("x-api-key", "client-key")
	req.Header.Set("anthropic-version", "2023-06-01")
	req.Header.Set("anthropic-beta", "prompt-caching-2024-07-31")
	req.Header.Set("Expect", "100-continue")
	rec := httptest.NewRecorder()
	ph.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", rec.Code)
	}
	if captured == nil || captured.URL.Path != "/v1/chat/completions" {
		t.Fatalf("upstream path = %v, want /v1/chat/completions", captured.URL)
	}
	for _, header := range []string{"anthropic-version", "anthropic-beta", "x-api-key", "Expect"} {
		if got := captured.Header.Get(header); got != "" {
			t.Fatalf("upstream %s header leaked: %q", header, got)
		}
	}
	if got := captured.Header.Get("Accept"); got != "text/event-stream" {
		t.Fatalf("upstream Accept = %q, want text/event-stream", got)
	}
	if got := captured.Header.Get("User-Agent"); got != "curl/8.21.0" {
		t.Fatalf("upstream User-Agent = %q, want low-latency fallback", got)
	}
	var body map[string]interface{}
	if err := json.Unmarshal(capturedBody, &body); err != nil {
		t.Fatalf("upstream body is invalid JSON: %v", err)
	}
	if body["model"] != "z-ai/glm-5.3-free" {
		t.Fatalf("normalized body = %#v, want GLM model", body)
	}
	// The caller sent no effort hint, so this pool's default (low) is
	// injected — an explicit hint would pass through unchanged.
	if got, _ := body["reasoning_effort"].(string); got != "low" {
		t.Fatalf("unspecified request effort = %q, want pool default low", got)
	}
}

func TestProxyAnthropicBridgeDoesNotFabricateCompletionOnTruncatedStream(t *testing.T) {
	ph := NewProxyHandler(testPool(t, "https://api.tokenrouter.com/v1", "", false))
	ph.client.Transport = roundTripFunc(func(req *http.Request) (*http.Response, error) {
		return &http.Response{
			StatusCode: http.StatusOK,
			Header:     http.Header{"Content-Type": []string{"text/event-stream"}},
			Body: io.NopCloser(strings.NewReader(
				"data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"partial thinking\"}}]}\n\n",
			)),
			Request: req,
		}, nil
	})

	req := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(`{"model":"glm-5.3-free","stream":true,"messages":[]}`))
	rec := httptest.NewRecorder()
	ph.ServeHTTP(rec, req)

	if strings.Contains(rec.Body.String(), "event: message_stop") {
		t.Fatal("truncated upstream stream was presented as a normal completion")
	}
	logs := ph.GetLogs()
	if len(logs) != 1 || logs[0].StatusCode != http.StatusBadGateway || !strings.Contains(logs[0].ErrorMsg, "before completion") {
		t.Fatalf("truncated stream log = %#v, want diagnostic 502", logs)
	}
}

func TestProxyRetriesEmptyAnthropicStreamBeforeVisibleOutput(t *testing.T) {
	ph := NewProxyHandler(testPool(t, "https://api.tokenrouter.com/v1", "", false))
	ph.pool.keys["k2"] = &KeyItem{Config: KeyConfig{ID: "k2", Name: "K2", Key: "test-key-2", RPMLimit: 10, Enabled: true}, Timestamps: make([]time.Time, 0)}
	ph.pool.keyOrder = append(ph.pool.keyOrder, "k2")
	var calls int
	ph.client.Transport = roundTripFunc(func(req *http.Request) (*http.Response, error) {
		calls++
		body := "data: {\"id\":\"empty\",\"choices\":[]}\n\n"
		if calls == 2 {
			body = "data: {\"id\":\"ok\",\"choices\":[{\"delta\":{\"content\":\"OK\"}}]}\n\ndata: [DONE]\n\n"
		}
		return &http.Response{
			StatusCode: http.StatusOK,
			Header:     http.Header{"Content-Type": []string{"text/event-stream"}},
			Body:       io.NopCloser(strings.NewReader(body)),
			Request:    req,
		}, nil
	})

	req := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(`{"model":"glm-5.3-free","stream":true,"messages":[]}`))
	rec := httptest.NewRecorder()
	ph.ServeHTTP(rec, req)

	if calls != 2 || rec.Code != http.StatusOK {
		t.Fatalf("calls=%d status=%d, want one safe retry and 200", calls, rec.Code)
	}
	body := rec.Body.String()
	if got := strings.Count(body, "event: message_start"); got != 1 {
		t.Fatalf("message_start count = %d, want exactly one after usable upstream output", got)
	}
	if !strings.Contains(body, `"text":"OK"`) || !strings.Contains(body, "event: message_stop") {
		t.Fatalf("retried stream was incomplete: %q", body)
	}
	logs := ph.GetLogs()
	if len(logs) != 1 || logs[0].StatusCode != http.StatusOK || logs[0].Attempts != 2 {
		t.Fatalf("retry log = %#v, want one successful two-attempt record", logs)
	}
}

func TestProxyRetryUsesFreshRequestContext(t *testing.T) {
	ph := NewProxyHandler(testPool(t, "https://api.tokenrouter.com/v1", "", false))
	ph.pool.keys["k2"] = &KeyItem{Config: KeyConfig{ID: "k2", Name: "K2", Key: "test-key-2", RPMLimit: 1, Enabled: true}, Timestamps: make([]time.Time, 0)}
	ph.pool.keyOrder = append(ph.pool.keyOrder, "k2")
	var calls int
	ph.client.Transport = roundTripFunc(func(req *http.Request) (*http.Response, error) {
		calls++
		if calls == 1 {
			return nil, context.Canceled
		}
		if err := req.Context().Err(); err != nil {
			return nil, err
		}
		return &http.Response{
			StatusCode: http.StatusOK,
			Header:     http.Header{"Content-Type": []string{"application/json"}},
			Body:       io.NopCloser(strings.NewReader(`{"ok":true}`)),
			Request:    req,
		}, nil
	})
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"z-ai/glm-5.3-free","messages":[]}`))
	rec := httptest.NewRecorder()
	ph.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK || calls != 2 {
		t.Fatalf("status=%d calls=%d, want 200 after one alternate-key retry", rec.Code, calls)
	}
}

func TestProxyRoutesFlashToFreebuffWithoutLeakingKey(t *testing.T) {
	var receivedAuth string
	freebuff := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		receivedAuth = r.Header.Get("Authorization")
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"ok":true}`))
	}))
	defer freebuff.Close()
	upstream := httptest.NewServer(http.NotFoundHandler())
	defer upstream.Close()

	ph := NewProxyHandler(testPool(t, upstream.URL, freebuff.URL, true))
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"z-ai/glm-5.3-flash"}`))
	rec := httptest.NewRecorder()
	ph.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK || rec.Body.String() != `{"ok":true}` {
		t.Fatalf("response = %d %q, want Freebuff response", rec.Code, rec.Body.String())
	}
	if receivedAuth != "" {
		t.Fatalf("Freebuff received leaked Authorization header %q", receivedAuth)
	}
	if got := ph.pool.keys["k1"].TotalRequests; got != 0 {
		t.Fatalf("Freebuff request consumed TokenRouter key slots: %d", got)
	}
}

func TestProxyPreservesThinkingSSEBytes(t *testing.T) {
	const event = "event: content_block_delta\ndata: {\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"keep this\"}}\n\n"
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, event)
	}))
	defer upstream.Close()

	ph := NewProxyHandler(testPool(t, upstream.URL, "", false))
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(`{"model":"glm-5.3-free","stream":true}`))
	rec := httptest.NewRecorder()
	ph.ServeHTTP(rec, req)

	body := rec.Body.String()
	// Keep-alive comments precede the first event, and a stream that ends
	// without a terminal event gets a diagnostic SSE error appended. The
	// thinking event itself must pass through byte-for-byte.
	if !strings.HasPrefix(body, ": ping\n\n") {
		t.Fatalf("SSE body missing keep-alive prefix: %q", body)
	}
	if !strings.Contains(body, event) {
		t.Fatalf("SSE body lost original event bytes: %q", body)
	}
	if strings.Count(body, "event: error") != 1 || !strings.Contains(body, "upstream stream ended before completion") {
		t.Fatalf("truncated stream missing diagnostic error: %q", body)
	}
}

func TestProxyFlushesPartialSSEChunkImmediately(t *testing.T) {
	releaseUpstream := make(chan struct{})
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, ok := w.(http.Flusher)
		if !ok {
			t.Fatal("upstream ResponseWriter does not support Flush")
		}
		_, _ = io.WriteString(w, "data: partial")
		flusher.Flush()
		<-releaseUpstream
		_, _ = io.WriteString(w, "\n\n")
		flusher.Flush()
	}))
	defer upstream.Close()

	ph := NewProxyHandler(testPool(t, upstream.URL, "", false))
	rec := &flushSignalRecorder{ResponseRecorder: httptest.NewRecorder(), firstWrite: make(chan struct{}, 1)}
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(`{"model":"glm-5.3-free","stream":true}`))
	done := make(chan struct{})
	go func() {
		ph.ServeHTTP(rec, req)
		close(done)
	}()

	select {
	case <-rec.firstWrite:
		// The raw SSE path must forward a partial upstream read before a newline.
	case <-time.After(750 * time.Millisecond):
		close(releaseUpstream)
		t.Fatal("proxy buffered a partial SSE chunk until the newline arrived")
	}
	close(releaseUpstream)
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("proxy did not finish after upstream released the stream")
	}
	body := rec.Body.String()
	if !strings.HasPrefix(body, ": ping\n\n") {
		t.Fatalf("SSE body missing keep-alive prefix: %q", body)
	}
	if !strings.Contains(body, "data: partial\n\n") {
		t.Fatalf("SSE body missing partial chunk: %q", body)
	}
	if strings.Count(body, "event: error") != 1 || !strings.Contains(body, "upstream stream ended before completion") {
		t.Fatalf("truncated partial stream missing diagnostic: %q", body)
	}
}

type flushSignalRecorder struct {
	*httptest.ResponseRecorder
	firstWrite chan struct{}
}

func (r *flushSignalRecorder) Write(p []byte) (int, error) {
	select {
	case r.firstWrite <- struct{}{}:
	default:
	}
	return r.ResponseRecorder.Write(p)
}

func (r *flushSignalRecorder) Flush() {
	r.ResponseRecorder.Flush()
}

func TestProxyInjectsDefaultEffortOnlyWhenCallerUnspecified(t *testing.T) {
	var receivedA, receivedB map[string]interface{}
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var parsed map[string]interface{}
		_ = json.Unmarshal(body, &parsed)
		if receivedA == nil {
			receivedA = parsed
		} else {
			receivedB = parsed
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"ok":true}`)
	}))
	defer upstream.Close()

	ph := NewProxyHandler(testPool(t, upstream.URL, "", false))
	ph.pool.keys["k1"].Config.RPMLimit = 10
	req1 := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"z-ai/glm-5.3-free","messages":[]}`))
	rec1 := httptest.NewRecorder()
	ph.ServeHTTP(rec1, req1)
	req2 := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"z-ai/glm-5.3-free","reasoning_effort":"low","messages":[]}`))
	rec2 := httptest.NewRecorder()
	ph.ServeHTTP(rec2, req2)

	if rec1.Code != http.StatusOK || rec2.Code != http.StatusOK {
		t.Fatalf("statuses = %d/%d, want 200/200", rec1.Code, rec2.Code)
	}
	// Unspecified request inherits the pool default; GLM shortens its pre-
	// content reasoning phase when given an explicit hint.
	if got, _ := receivedA["reasoning_effort"].(string); got != "max" {
		t.Fatalf("default effort = %q, want pool default max injected", got)
	}
	// An explicit caller effort is caller-owned and must pass through as-is.
	if got, _ := receivedB["reasoning_effort"].(string); got != "low" {
		t.Fatalf("explicit effort = %q, want caller value low preserved", got)
	}
}

func TestProxyPreservesExplicitGLMReasoningEffort(t *testing.T) {
	var received map[string]interface{}
	var receivedUA string
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer r.Body.Close()
		receivedUA = r.Header.Get("User-Agent")
		if err := json.NewDecoder(r.Body).Decode(&received); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"ok":true}`)
	}))
	defer upstream.Close()

	ph := NewProxyHandler(testPool(t, upstream.URL, "", false))
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"z-ai/glm-5.3-free","reasoning_effort":"max","messages":[]}`))
	rec := httptest.NewRecorder()
	ph.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", rec.Code)
	}
	if got, _ := received["reasoning_effort"].(string); got != "max" {
		t.Fatalf("reasoning_effort = %q, want caller value max", got)
	}
	if receivedUA != "curl/8.21.0" {
		t.Fatalf("upstream User-Agent = %q, want low-latency fallback", receivedUA)
	}
}
