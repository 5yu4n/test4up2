package router

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

// TokenUsageStats is the dashboard-facing cumulative token counter. Provider
// usage fields are used when available; requests without those fields are
// marked estimated so the UI never presents a byte-based fallback as exact.
type TokenUsageStats struct {
	InputTokens       int64                      `json:"input_tokens"`
	OutputTokens      int64                      `json:"output_tokens"`
	TotalTokens       int64                      `json:"total_tokens"`
	Requests          int64                      `json:"requests"`
	ExactRequests     int64                      `json:"exact_requests"`
	EstimatedRequests int64                      `json:"estimated_requests"`
	StartedAt         time.Time                  `json:"started_at"`
	LastUpdated       time.Time                  `json:"last_updated"`
	ByKey             map[string]TokenUsageStats `json:"by_key,omitempty"`
}

type tokenUsageMeasurement struct {
	InputTokens  int64
	OutputTokens int64
	Estimated    bool
}

type parsedTokenUsage struct {
	InputTokens  int64
	OutputTokens int64
	InputKnown   bool
	OutputKnown  bool
}

type tokenUsageState struct {
	Version int             `json:"version"`
	Stats   TokenUsageStats `json:"stats"`
}

type tokenUsageStore struct {
	mu     sync.RWMutex
	saveMu sync.Mutex
	path   string
	state  tokenUsageState
}

func newTokenUsageStore(path string, historical []RequestLogItem) *tokenUsageStore {
	if path == "" {
		path = "tokenrouter-usage.json"
	}
	s := &tokenUsageStore{
		path: path,
		state: tokenUsageState{
			Version: 1,
			Stats: TokenUsageStats{
				StartedAt: time.Now(),
				ByKey:     make(map[string]TokenUsageStats),
			},
		},
	}

	loaded := false
	if data, err := os.ReadFile(path); err == nil {
		if json.Unmarshal(data, &s.state) == nil {
			loaded = true
			if s.state.Version == 0 {
				s.state.Version = 1
			}
			if s.state.Stats.StartedAt.IsZero() {
				s.state.Stats.StartedAt = time.Now()
			}
			if s.state.Stats.ByKey == nil {
				s.state.Stats.ByKey = make(map[string]TokenUsageStats)
			}
		}
	}
	if !loaded {
		// Migrate any retained request history. Older records did not carry
		// provider usage, so these values are deliberately marked estimated.
		var firstHistoricalTimestamp time.Time
		for _, item := range historical {
			if item.StatusCode < 200 || item.StatusCode >= 300 || !isTokenUsagePath(item.Path) {
				continue
			}
			measurement := measurementFromRequestLog(item)
			s.addLocked(item.KeyID, measurement)
			if !item.Timestamp.IsZero() && (firstHistoricalTimestamp.IsZero() || item.Timestamp.Before(firstHistoricalTimestamp)) {
				firstHistoricalTimestamp = item.Timestamp
			}
		}
		if !firstHistoricalTimestamp.IsZero() {
			s.state.Stats.StartedAt = firstHistoricalTimestamp
			for key, stats := range s.state.Stats.ByKey {
				stats.StartedAt = firstHistoricalTimestamp
				s.state.Stats.ByKey[key] = stats
			}
		}
		// Persist the initial/migrated snapshot before returning. Besides making
		// a restart immediately durable, this avoids leaving a background write
		// racing process shutdown when a short-lived test or utility constructs a
		// handler and exits right away.
		_ = s.saveNow()
	}
	return s
}

func (s *tokenUsageStore) saveNow() error {
	if s == nil || s.path == "" {
		return nil
	}
	s.saveMu.Lock()
	defer s.saveMu.Unlock()
	s.mu.RLock()
	data, err := json.MarshalIndent(s.state, "", "  ")
	s.mu.RUnlock()
	if err != nil {
		return err
	}
	if dir := filepath.Dir(s.path); dir != "." && dir != "" {
		if err := os.MkdirAll(dir, 0o700); err != nil {
			return err
		}
	}
	tmp := s.path + ".tmp"
	if err := os.WriteFile(tmp, data, 0o600); err != nil {
		return err
	}
	if err := os.Rename(tmp, s.path); err != nil {
		// Windows cannot replace an existing file with Rename. Remove only the
		// known target and retry; the temporary file remains private to this
		// store instance.
		_ = os.Remove(s.path)
		if retryErr := os.Rename(tmp, s.path); retryErr != nil {
			_ = os.Remove(tmp)
			return retryErr
		}
	}
	return nil
}

func (s *tokenUsageStore) add(keyID string, measurement tokenUsageMeasurement) {
	if s == nil || measurement.InputTokens < 0 || measurement.OutputTokens < 0 {
		return
	}
	s.mu.Lock()
	s.addLocked(keyID, measurement)
	s.mu.Unlock()
	// The file is tiny and writes happen after a response has been fully
	// received. Persist synchronously so a process restart cannot lose the
	// latest counter and so no background writer can race deployment/tests.
	_ = s.saveNow()
}

func (s *tokenUsageStore) addLocked(keyID string, measurement tokenUsageMeasurement) {
	now := time.Now()
	stats := &s.state.Stats
	stats.InputTokens += measurement.InputTokens
	stats.OutputTokens += measurement.OutputTokens
	stats.TotalTokens += measurement.InputTokens + measurement.OutputTokens
	stats.Requests++
	if measurement.Estimated {
		stats.EstimatedRequests++
	} else {
		stats.ExactRequests++
	}
	if stats.StartedAt.IsZero() {
		stats.StartedAt = now
	}
	stats.LastUpdated = now
	if stats.ByKey == nil {
		stats.ByKey = make(map[string]TokenUsageStats)
	}
	keyStats := stats.ByKey[keyID]
	keyStats.InputTokens += measurement.InputTokens
	keyStats.OutputTokens += measurement.OutputTokens
	keyStats.TotalTokens += measurement.InputTokens + measurement.OutputTokens
	keyStats.Requests++
	if measurement.Estimated {
		keyStats.EstimatedRequests++
	} else {
		keyStats.ExactRequests++
	}
	keyStats.StartedAt = stats.StartedAt
	keyStats.LastUpdated = now
	stats.ByKey[keyID] = keyStats
}

func (s *tokenUsageStore) snapshot() TokenUsageStats {
	if s == nil {
		return TokenUsageStats{ByKey: make(map[string]TokenUsageStats)}
	}
	s.mu.RLock()
	result := s.state.Stats
	result.ByKey = make(map[string]TokenUsageStats, len(s.state.Stats.ByKey))
	for key, stats := range s.state.Stats.ByKey {
		stats.ByKey = nil
		result.ByKey[key] = stats
	}
	s.mu.RUnlock()
	return result
}

func measurementFromRequestLog(item RequestLogItem) tokenUsageMeasurement {
	input := item.InputTokens
	output := item.OutputTokens
	estimated := item.TokensEstimated
	if input <= 0 {
		input = estimateTokensFromBytes(item.RequestBytes)
		estimated = true
	}
	if output <= 0 {
		output = estimateTokensFromBytes(item.BytesOut)
		if item.BytesOut > 0 {
			estimated = true
		}
	}
	return tokenUsageMeasurement{InputTokens: input, OutputTokens: output, Estimated: estimated}
}

func measurementFromPayload(inputBody []byte, outputBytes int64, usage parsedTokenUsage) tokenUsageMeasurement {
	input := usage.InputTokens
	output := usage.OutputTokens
	estimated := !usage.InputKnown || !usage.OutputKnown
	if !usage.InputKnown {
		input = estimateTokensFromBytes(int64(len(inputBody)))
	}
	if !usage.OutputKnown {
		output = estimateTokensFromBytes(outputBytes)
	}
	return tokenUsageMeasurement{InputTokens: input, OutputTokens: output, Estimated: estimated}
}

func estimateTokensFromBytes(size int64) int64 {
	if size <= 0 {
		return 0
	}
	return (size + 3) / 4
}

func parseTokenUsage(body []byte) parsedTokenUsage {
	if len(body) == 0 {
		return parsedTokenUsage{}
	}
	var envelope struct {
		Usage map[string]interface{} `json:"usage"`
	}
	if json.Unmarshal(body, &envelope) != nil {
		return parsedTokenUsage{}
	}
	return parseTokenUsageMap(envelope.Usage)
}

func parseTokenUsageMap(usage map[string]interface{}) parsedTokenUsage {
	if len(usage) == 0 {
		return parsedTokenUsage{}
	}
	result := parsedTokenUsage{}
	for _, key := range []string{"prompt_tokens", "input_tokens", "promptTokens", "inputTokens"} {
		if value, ok := numberValue(usage[key]); ok {
			result.InputTokens = value
			result.InputKnown = true
			break
		}
	}
	for _, key := range []string{"completion_tokens", "output_tokens", "completionTokens", "outputTokens"} {
		if value, ok := numberValue(usage[key]); ok {
			result.OutputTokens = value
			result.OutputKnown = true
			break
		}
	}
	return result
}

func numberValue(value interface{}) (int64, bool) {
	switch typed := value.(type) {
	case float64:
		if typed >= 0 {
			return int64(typed), true
		}
	case float32:
		if typed >= 0 {
			return int64(typed), true
		}
	case int:
		if typed >= 0 {
			return int64(typed), true
		}
	case int64:
		if typed >= 0 {
			return typed, true
		}
	case json.Number:
		if parsed, err := typed.Int64(); err == nil && parsed >= 0 {
			return parsed, true
		}
	}
	return 0, false
}

func mergeTokenUsage(dst *parsedTokenUsage, src parsedTokenUsage) {
	if dst == nil {
		return
	}
	if src.InputKnown {
		dst.InputTokens = src.InputTokens
		dst.InputKnown = true
	}
	if src.OutputKnown {
		dst.OutputTokens = src.OutputTokens
		dst.OutputKnown = true
	}
}

func isTokenUsagePath(path string) bool {
	path = strings.ToLower(path)
	return strings.Contains(path, "/messages") || strings.Contains(path, "/chat/completions") || strings.Contains(path, "/embeddings")
}
