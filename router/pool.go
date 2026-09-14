package router

import (
	"context"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"sort"
	"strings"
	"sync"
	"time"
)

type KeyConfig struct {
	ID       string `json:"id"`
	Name     string `json:"name"`
	Key      string `json:"key"`
	RPMLimit int    `json:"rpm_limit"`
	Enabled  bool   `json:"enabled"`
}

type Config struct {
	Port                     int               `json:"port"`
	UpstreamURL              string            `json:"upstream_url"`
	FreebuffBaseURL          string            `json:"freebuff_base_url"`
	FreebuffEnabled          bool              `json:"freebuff_enabled"`
	MaxQueueSeconds          int               `json:"max_queue_seconds"`
	MaxConcurrentRequests    int               `json:"max_concurrent_requests"`
	StripThinking            bool              `json:"strip_thinking"`
	FastStreamThinkingAsText bool              `json:"fast_stream_thinking_as_text"`
	DefaultEffort            string            `json:"default_effort"`
	SSEPingIntervalMs        int               `json:"sse_ping_interval_ms"`
	ModelMappings            map[string]string `json:"model_mappings"`
	AllowedModels            []string          `json:"allowed_models"`
	DashboardPasswordSalt    string            `json:"dashboard_password_salt,omitempty"`
	DashboardPasswordHash    string            `json:"dashboard_password_hash,omitempty"`
	Keys                     []KeyConfig       `json:"keys"`
}

type KeyItem struct {
	Config                    KeyConfig
	Timestamps                []time.Time
	TotalRequests             int64
	SuccessCount              int64
	ErrorCount                int64
	ConsecutiveErrors         int64
	ConsecutiveHeaderTimeouts int64
	InFlightRequests          int64
	LastUsedTime              time.Time
	CooldownUntil             time.Time
	LatencyEMAMs              int64
	LatencySamples            int64
	GapEMAMs                  int64
	GapSamples                int64
}

type keyMetricsSnapshot struct {
	LatencyEMAMs              int64 `json:"latency_ema_ms,omitempty"`
	LatencySamples            int64 `json:"latency_samples,omitempty"`
	GapEMAMs                  int64 `json:"gap_ema_ms,omitempty"`
	GapSamples                int64 `json:"gap_samples,omitempty"`
	ConsecutiveHeaderTimeouts int64 `json:"consecutive_header_timeouts,omitempty"`
	CooldownUntilUnixMs       int64 `json:"cooldown_until_unix_ms,omitempty"`
}

func keySelectionScore(item *KeyItem) int64 {
	if item == nil {
		return 1<<62 - 1
	}
	// Unobserved keys get a low exploration score so every configured key is
	// sampled at least once. Once a key has data, rank it by observed first-byte
	// latency. A 5s penalty per in-flight request keeps fast keys from being
	// overloaded while the pool learns the current upstream conditions.
	score := int64(2000)
	if item.LatencySamples > 0 && item.LatencyEMAMs > 0 {
		score = item.LatencyEMAMs
	}
	if item.GapSamples > 0 {
		// Upstream stalls are directly visible as frozen output, so give the
		// observed inter-chunk gap the same weight as first-byte latency.
		score += item.GapEMAMs
	}
	if item.ConsecutiveErrors > 0 {
		// Keep a key that recently produced 4xx/5xx responses out of the hot
		// path long enough for the other keys to absorb traffic. Cap the penalty
		// so a transient failure never permanently disables a key.
		errors := item.ConsecutiveErrors
		if errors > 3 {
			errors = 3
		}
		score += errors * 5000
	}
	return score + item.InFlightRequests*5000
}

type KeyStatusDTO struct {
	ID                   string    `json:"id"`
	Name                 string    `json:"name"`
	KeyMasked            string    `json:"key_masked"`
	RPMLimit             int       `json:"rpm_limit"`
	RPMCurrent           int       `json:"rpm_current"`
	Enabled              bool      `json:"enabled"`
	TotalRequests        int64     `json:"total_requests"`
	SuccessCount         int64     `json:"success_count"`
	ErrorCount           int64     `json:"error_count"`
	InFlightRequests     int64     `json:"in_flight_requests"`
	LastUsedTime         time.Time `json:"last_used_time"`
	InCooldown           bool      `json:"in_cooldown"`
	Eligible             bool      `json:"eligible"`
	Busy                 bool      `json:"busy"`
	RPMExhausted         bool      `json:"rpm_exhausted"`
	CooldownRemainingSec int       `json:"cooldown_remaining_sec"`
	NextWindowSec        int       `json:"next_window_sec"`
	LatencyEMAMs         int64     `json:"latency_ema_ms,omitempty"`
	LatencySamples       int64     `json:"latency_samples,omitempty"`
	GapEMAMs             int64     `json:"gap_ema_ms,omitempty"`
	GapSamples           int64     `json:"gap_samples,omitempty"`
}

type PoolStats struct {
	UpstreamURL              string            `json:"upstream_url"`
	FreebuffBaseURL          string            `json:"freebuff_base_url"`
	FreebuffEnabled          bool              `json:"freebuff_enabled"`
	Port                     int               `json:"port"`
	TotalKeys                int               `json:"total_keys"`
	ActiveKeys               int               `json:"active_keys"`
	EligibleKeys             int               `json:"eligible_keys"`
	BusyKeys                 int               `json:"busy_keys"`
	CooldownKeys             int               `json:"cooldown_keys"`
	RPMExhaustedKeys         int               `json:"rpm_exhausted_keys"`
	AggregateMaxRPM          int               `json:"aggregate_max_rpm"`
	AggregateCurrRPM         int               `json:"aggregate_curr_rpm"`
	TotalRequests            int64             `json:"total_requests"`
	QueuedRequests           int               `json:"queued_requests"`
	QueueBlockReason         string            `json:"queue_block_reason,omitempty"`
	ActiveRequests           int               `json:"active_requests"`
	MaxConcurrentRequests    int               `json:"max_concurrent_requests"`
	EffectiveMaxConcurrent   int               `json:"effective_max_concurrent_requests"`
	ConcurrencyBackoffSec    int               `json:"concurrency_backoff_remaining_sec,omitempty"`
	StripThinking            bool              `json:"strip_thinking"`
	FastStreamThinkingAsText bool              `json:"fast_stream_thinking_as_text"`
	DefaultEffort            string            `json:"default_effort"`
	SSEPingIntervalMs        int               `json:"sse_ping_interval_ms"`
	ModelMappings            map[string]string `json:"model_mappings"`
	AllowedModels            []string          `json:"allowed_models"`
	TokenUsage               TokenUsageStats   `json:"token_usage"`
	Keys                     []KeyStatusDTO    `json:"keys"`
}

type Pool struct {
	mu                       sync.RWMutex
	metricsWriteMu           sync.Mutex
	cond                     *sync.Cond
	configPath               string
	metricsPath              string
	port                     int
	upstreamURL              string
	freebuffBaseURL          string
	freebuffEnabled          bool
	maxQueueSeconds          int
	maxConcurrentRequests    int
	activeRequests           int
	stripThinking            bool
	fastStreamThinkingAsText bool
	defaultEffort            string
	ssePingIntervalMs        int
	modelMappings            map[string]string
	allowedModels            map[string]struct{}
	dashboardPasswordSalt    string
	dashboardPasswordHash    string
	keys                     map[string]*KeyItem
	keyOrder                 []string
	queuedCount              int
	rrIndex                  uint64
	concurrencyBackoffUntil  time.Time
	concurrencyBackoffLimit  int
	concurrencyRecoveryEvery time.Duration
	rateLimitSignalWindow    time.Duration
	rateLimitSignals         map[string]time.Time
	lastConcurrencyDecrease  time.Time
}

const defaultRateLimitSignalWindow = 2 * time.Second

func NewPool(configPath string) (*Pool, error) {
	p := &Pool{
		configPath:            configPath,
		metricsPath:           configPath + ".metrics.json",
		keys:                  make(map[string]*KeyItem),
		rateLimitSignalWindow: defaultRateLimitSignalWindow,
		rateLimitSignals:      make(map[string]time.Time),
	}
	p.cond = sync.NewCond(&p.mu)

	if err := p.LoadConfig(); err != nil {
		return nil, err
	}

	// Start background cleanup ticker for sliding windows
	go p.backgroundCleaner()

	return p, nil
}

func (p *Pool) LoadConfig() error {
	p.mu.Lock()
	defer p.mu.Unlock()

	data, err := os.ReadFile(p.configPath)
	if err != nil {
		return fmt.Errorf("failed to read config file: %w", err)
	}

	var cfg Config
	if err := json.Unmarshal(data, &cfg); err != nil {
		return fmt.Errorf("failed to parse config json: %w", err)
	}

	p.port = cfg.Port
	if p.port == 0 {
		p.port = 8080
	}
	p.upstreamURL = cfg.UpstreamURL
	if p.upstreamURL == "" {
		p.upstreamURL = "https://api.openai.com"
	}
	p.freebuffBaseURL = cfg.FreebuffBaseURL
	if p.freebuffBaseURL == "" {
		p.freebuffBaseURL = "http://127.0.0.1:8090/v1"
	}
	p.freebuffEnabled = cfg.FreebuffEnabled
	p.maxQueueSeconds = cfg.MaxQueueSeconds
	if p.maxQueueSeconds <= 0 {
		p.maxQueueSeconds = 60
	}
	p.maxConcurrentRequests = cfg.MaxConcurrentRequests

	p.stripThinking = cfg.StripThinking
	p.fastStreamThinkingAsText = cfg.FastStreamThinkingAsText
	p.defaultEffort = cfg.DefaultEffort
	if p.defaultEffort == "" {
		p.defaultEffort = "low"
	}
	p.ssePingIntervalMs = cfg.SSEPingIntervalMs
	if p.ssePingIntervalMs <= 0 {
		p.ssePingIntervalMs = 500
	}
	p.modelMappings = cfg.ModelMappings
	if p.modelMappings == nil {
		p.modelMappings = make(map[string]string)
	}
	p.allowedModels = make(map[string]struct{}, len(cfg.AllowedModels))
	for _, model := range cfg.AllowedModels {
		model = strings.TrimSpace(model)
		if model != "" {
			p.allowedModels[model] = struct{}{}
		}
	}
	p.dashboardPasswordSalt = strings.TrimSpace(cfg.DashboardPasswordSalt)
	p.dashboardPasswordHash = strings.TrimSpace(cfg.DashboardPasswordHash)

	p.keyOrder = make([]string, 0, len(cfg.Keys))
	for _, kcfg := range cfg.Keys {
		p.keyOrder = append(p.keyOrder, kcfg.ID)
		if existing, exists := p.keys[kcfg.ID]; exists {
			existing.Config = kcfg
		} else {
			p.keys[kcfg.ID] = &KeyItem{
				Config:     kcfg,
				Timestamps: make([]time.Time, 0),
			}
		}
	}
	p.loadMetricsLocked()

	return nil
}

func (p *Pool) loadMetricsLocked() {
	data, err := os.ReadFile(p.metricsPath)
	if err != nil {
		return
	}
	var snapshots map[string]keyMetricsSnapshot
	if err := json.Unmarshal(data, &snapshots); err != nil {
		return
	}
	for id, snapshot := range snapshots {
		if item, ok := p.keys[id]; ok {
			item.LatencyEMAMs = snapshot.LatencyEMAMs
			item.LatencySamples = snapshot.LatencySamples
			item.GapEMAMs = snapshot.GapEMAMs
			item.GapSamples = snapshot.GapSamples
			// Header-timeout benches survive restarts so a pool relaunch does
			// not relearn dead keys by burning a full budget on each one.
			item.ConsecutiveHeaderTimeouts = snapshot.ConsecutiveHeaderTimeouts
			if snapshot.CooldownUntilUnixMs > time.Now().UnixMilli() {
				item.CooldownUntil = time.UnixMilli(snapshot.CooldownUntilUnixMs)
			}
		}
	}
}

func (p *Pool) saveMetrics() error {
	p.metricsWriteMu.Lock()
	defer p.metricsWriteMu.Unlock()

	p.mu.RLock()
	snapshots := make(map[string]keyMetricsSnapshot, len(p.keys))
	for id, item := range p.keys {
		snapshot := keyMetricsSnapshot{
			LatencyEMAMs:              item.LatencyEMAMs,
			LatencySamples:            item.LatencySamples,
			GapEMAMs:                  item.GapEMAMs,
			GapSamples:                item.GapSamples,
			ConsecutiveHeaderTimeouts: item.ConsecutiveHeaderTimeouts,
		}
		if item.CooldownUntil.After(time.Now()) {
			snapshot.CooldownUntilUnixMs = item.CooldownUntil.UnixMilli()
		}
		snapshots[id] = snapshot
	}
	path := p.metricsPath
	p.mu.RUnlock()

	data, err := json.MarshalIndent(snapshots, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0600)
}

func (p *Pool) SaveConfig() error {
	p.mu.RLock()
	cfg := Config{
		Port:                     p.port,
		UpstreamURL:              p.upstreamURL,
		FreebuffBaseURL:          p.freebuffBaseURL,
		FreebuffEnabled:          p.freebuffEnabled,
		MaxQueueSeconds:          p.maxQueueSeconds,
		MaxConcurrentRequests:    p.maxConcurrentRequests,
		StripThinking:            p.stripThinking,
		FastStreamThinkingAsText: p.fastStreamThinkingAsText,
		DefaultEffort:            p.defaultEffort,
		SSEPingIntervalMs:        p.ssePingIntervalMs,
		ModelMappings:            p.modelMappings,
		AllowedModels:            make([]string, 0, len(p.allowedModels)),
		DashboardPasswordSalt:    p.dashboardPasswordSalt,
		DashboardPasswordHash:    p.dashboardPasswordHash,
		Keys:                     make([]KeyConfig, 0, len(p.keyOrder)),
	}
	for model := range p.allowedModels {
		cfg.AllowedModels = append(cfg.AllowedModels, model)
	}
	sort.Strings(cfg.AllowedModels)
	for _, id := range p.keyOrder {
		if k, ok := p.keys[id]; ok {
			cfg.Keys = append(cfg.Keys, k.Config)
		}
	}
	p.mu.RUnlock()

	data, err := json.MarshalIndent(cfg, "", "  ")
	if err != nil {
		return err
	}

	return os.WriteFile(p.configPath, data, 0600)
}

// VerifyDashboardPassword checks the configured dashboard password without
// exposing the password or its digest to callers. The password is generated
// with high entropy during deployment; the salted SHA-256 digest is sufficient
// here because the password is not user-chosen.
func (p *Pool) VerifyDashboardPassword(password string) bool {
	p.mu.RLock()
	salt := p.dashboardPasswordSalt
	expectedHex := p.dashboardPasswordHash
	p.mu.RUnlock()
	if salt == "" || expectedHex == "" {
		return false
	}
	expected, err := hex.DecodeString(expectedHex)
	if err != nil || len(expected) != sha256.Size {
		return false
	}
	actual := sha256.Sum256([]byte(salt + "\x00" + password))
	return subtle.ConstantTimeCompare(actual[:], expected) == 1
}

func (p *Pool) backgroundCleaner() {
	ticker := time.NewTicker(1 * time.Second)
	for range ticker.C {
		p.mu.Lock()
		now := time.Now()
		cutoff := now.Add(-60 * time.Second)
		changed := p.advanceConcurrencyRecoveryLocked(now)
		for _, item := range p.keys {
			oldLen := len(item.Timestamps)
			item.cleanTimestamps(cutoff)
			if len(item.Timestamps) < oldLen {
				changed = true
			}
		}
		if changed {
			p.cond.Broadcast()
		}
		p.mu.Unlock()
	}
}

func (k *KeyItem) cleanTimestamps(cutoff time.Time) {
	idx := 0
	for idx < len(k.Timestamps) && k.Timestamps[idx].Before(cutoff) {
		idx++
	}
	if idx > 0 {
		k.Timestamps = k.Timestamps[idx:]
	}
}

// AcquireKey gets an available key according to sliding window RPM limits.
// If no key is available, it waits smoothly in queue until a slot frees up or context is cancelled.
func (p *Pool) AcquireKey(ctx context.Context, excludeKeys map[string]bool) (*KeyItem, func(success bool), error) {
	return p.acquireKey(ctx, excludeKeys, 0)
}

// AcquireKeySized currently uses the configured account-wide concurrency
// ceiling for every request size. A previous two-slot large-request ceiling
// caused 256 KiB+ calls to queue for minutes even when all seven keys and nearly
// all RPM capacity were idle.
func (p *Pool) AcquireKeySized(ctx context.Context, excludeKeys map[string]bool, _ int64) (*KeyItem, func(success bool), error) {
	return p.acquireKey(ctx, excludeKeys, 0)
}

func (p *Pool) acquireKey(ctx context.Context, excludeKeys map[string]bool, concurrencyLimit int) (*KeyItem, func(success bool), error) {
	p.mu.Lock()
	defer p.mu.Unlock()

	// Fast-fail when every enabled key is already excluded for this request:
	// no amount of queueing can make one of them eligible again, so waiting
	// out the full queue deadline would only delay the client's retryable 429.
	if excludeKeys != nil {
		eligibleRemaining := 0
		for _, id := range p.keyOrder {
			item, ok := p.keys[id]
			if !ok || !item.Config.Enabled || excludeKeys[id] {
				continue
			}
			eligibleRemaining++
		}
		if eligibleRemaining == 0 {
			return nil, nil, errors.New("queue timeout: every enabled key failed for this request")
		}
	}

	p.queuedCount++
	defer func() {
		p.queuedCount--
	}()

	timeoutDuration := time.Duration(p.maxQueueSeconds) * time.Second
	deadline := time.Now().Add(timeoutDuration)

	for {
		now := time.Now()
		p.advanceConcurrencyRecoveryLocked(now)
		cutoff := now.Add(-60 * time.Second)
		var bestKey *KeyItem
		minRPM := 999999
		minInFlight := int64(1<<62 - 1)
		bestScore := int64(1<<62 - 1)
		oldestLastUsed := now
		blockReason := "all enabled keys are rate limited or cooling down"

		startIdx := 0
		if len(p.keyOrder) > 0 {
			startIdx = int(p.rrIndex % uint64(len(p.keyOrder)))
		}

		for i := 0; i < len(p.keyOrder); i++ {
			id := p.keyOrder[(startIdx+i)%len(p.keyOrder)]
			item, exists := p.keys[id]
			if !exists || !item.Config.Enabled {
				continue
			}
			if excludeKeys != nil && excludeKeys[item.Config.ID] {
				continue
			}
			if item.CooldownUntil.After(now) {
				continue
			}

			item.cleanTimestamps(cutoff)

			currentRPM := len(item.Timestamps)
			if currentRPM < item.Config.RPMLimit {
				candidateScore := keySelectionScore(item)
				// Spread load across independent keys before using latency as a
				// tie-breaker. Ranking only by historical latency repeatedly selected
				// the same three keys while four configured keys stayed untouched.
				isBetter := bestKey == nil || item.InFlightRequests < minInFlight ||
					(item.InFlightRequests == minInFlight && currentRPM < minRPM) ||
					(item.InFlightRequests == minInFlight && currentRPM == minRPM && candidateScore < bestScore) ||
					(item.InFlightRequests == minInFlight && currentRPM == minRPM && candidateScore == bestScore && item.LastUsedTime.Before(oldestLastUsed))

				if isBetter {
					bestKey = item
					minRPM = currentRPM
					minInFlight = item.InFlightRequests
					bestScore = candidateScore
					oldestLastUsed = item.LastUsedTime
				}
			}
		}
		// TokenRouter enforces an account-wide concurrency budget. Keep an
		// optional local ceiling so bursts wait in our short queue instead of
		// being queued or throttled after reaching the upstream gateway.
		maxConcurrent := p.effectiveMaxConcurrentLocked(now)
		if concurrencyLimit > 0 && (maxConcurrent <= 0 || concurrencyLimit < maxConcurrent) {
			maxConcurrent = concurrencyLimit
		}
		if maxConcurrent > 0 && p.activeRequests >= maxConcurrent {
			bestKey = nil
			blockReason = fmt.Sprintf("global concurrency saturated (%d/%d active)", p.activeRequests, maxConcurrent)
		}

		if bestKey != nil {
			p.rrIndex++
			// Record request in sliding window
			bestKey.Timestamps = append(bestKey.Timestamps, now)
			bestKey.InFlightRequests++
			p.activeRequests++
			bestKey.TotalRequests++
			bestKey.LastUsedTime = now

			keyCopy := *bestKey

			releaseFunc := func(success bool) {
				p.mu.Lock()
				defer p.mu.Unlock()

				if k, ok := p.keys[keyCopy.Config.ID]; ok {
					k.InFlightRequests--
					if k.InFlightRequests < 0 {
						k.InFlightRequests = 0
					}
					if success {
						k.SuccessCount++
						k.ConsecutiveErrors = 0
						k.ConsecutiveHeaderTimeouts = 0
					} else {
						k.ErrorCount++
						k.ConsecutiveErrors++
					}
				}
				if p.activeRequests > 0 {
					p.activeRequests--
				}
				p.cond.Broadcast()
			}

			return &keyCopy, releaseFunc, nil
		}

		// No key available right now. Check if context cancelled or queue deadline exceeded.
		if ctx.Err() != nil {
			return nil, nil, fmt.Errorf("queue canceled while %s: %w", blockReason, ctx.Err())
		}
		if time.Now().After(deadline) {
			return nil, nil, errors.New("queue timeout: " + blockReason)
		}

		// Calculate shortest wait duration until next sliding window slot opens
		waitDuration := p.calculateMinWaitDurationLocked(now)
		if p.concurrencyBackoffUntil.After(now) {
			recoveryWait := p.concurrencyBackoffUntil.Sub(now)
			if recoveryWait < waitDuration {
				waitDuration = recoveryWait
			}
		}
		if waitDuration <= 0 {
			waitDuration = 100 * time.Millisecond
		}
		if deadlineWait := time.Until(deadline); deadlineWait < waitDuration {
			waitDuration = deadlineWait
		}
		if waitDuration <= 0 {
			waitDuration = time.Millisecond
		}

		// Wait either on cond broadcast or timer
		timer := time.NewTimer(waitDuration)
		waitDone := make(chan struct{})

		go func() {
			shouldBroadcast := false
			select {
			case <-ctx.Done():
				shouldBroadcast = true
			case <-timer.C:
				shouldBroadcast = true
			case <-waitDone:
			}
			if shouldBroadcast {
				// Holding the condition lock prevents a cancellation or timer wakeup
				// from racing ahead of Cond.Wait and being lost.
				p.mu.Lock()
				p.cond.Broadcast()
				p.mu.Unlock()
			}
		}()

		p.cond.Wait()
		timer.Stop()
		close(waitDone)
	}
}

func (p *Pool) effectiveMaxConcurrentLocked(now time.Time) int {
	maxConcurrent := p.maxConcurrentRequests
	if p.concurrencyBackoffLimit > 0 && (maxConcurrent <= 0 || p.concurrencyBackoffLimit < maxConcurrent) {
		maxConcurrent = p.concurrencyBackoffLimit
	}
	return maxConcurrent
}

// advanceConcurrencyRecoveryLocked raises the AIMD ceiling one slot per
// recovery interval. The caller must hold p.mu for writing.
func (p *Pool) advanceConcurrencyRecoveryLocked(now time.Time) bool {
	if p.concurrencyBackoffLimit <= 0 || p.maxConcurrentRequests <= 0 {
		p.concurrencyBackoffLimit = 0
		p.concurrencyBackoffUntil = time.Time{}
		p.concurrencyRecoveryEvery = 0
		return false
	}
	if p.concurrencyBackoffLimit >= p.maxConcurrentRequests {
		p.concurrencyBackoffLimit = 0
		p.concurrencyBackoffUntil = time.Time{}
		p.concurrencyRecoveryEvery = 0
		return false
	}
	if p.concurrencyBackoffUntil.IsZero() || now.Before(p.concurrencyBackoffUntil) {
		return false
	}

	interval := p.concurrencyRecoveryEvery
	if interval <= 0 {
		interval = 5 * time.Second
	}
	steps := 1 + int(now.Sub(p.concurrencyBackoffUntil)/interval)
	p.concurrencyBackoffLimit += steps
	if p.concurrencyBackoffLimit >= p.maxConcurrentRequests {
		p.concurrencyBackoffLimit = 0
		p.concurrencyBackoffUntil = time.Time{}
		p.concurrencyRecoveryEvery = 0
	} else {
		p.concurrencyBackoffUntil = p.concurrencyBackoffUntil.Add(time.Duration(steps) * interval)
	}
	return true
}

// RecordUpstreamRateLimit records a per-key throttle signal. One key can be
// exhausted or unhealthy without proving an account-wide limit, so global
// concurrency is reduced only after two different enabled keys report a rate
// limit inside the short signal window. A confirmed account signal applies
// multiplicative decrease; capacity then returns additively, one slot per
// duration. Repeated signals from the same key never cause another decrease.
// The return value reports whether this call lowered global concurrency.
func (p *Pool) RecordUpstreamRateLimit(keyID string, duration time.Duration) bool {
	if duration <= 0 {
		duration = 5 * time.Second
	}
	p.mu.Lock()
	defer p.mu.Unlock()

	if p.maxConcurrentRequests <= 0 {
		return false
	}
	now := time.Now()
	p.advanceConcurrencyRecoveryLocked(now)

	item, exists := p.keys[keyID]
	if keyID == "" || !exists || !item.Config.Enabled {
		return false
	}
	window := p.rateLimitSignalWindow
	if window <= 0 {
		window = defaultRateLimitSignalWindow
	}
	cutoff := now.Add(-window)
	for id, seenAt := range p.rateLimitSignals {
		if seenAt.Before(cutoff) {
			delete(p.rateLimitSignals, id)
		}
	}
	if _, duplicate := p.rateLimitSignals[keyID]; duplicate {
		p.rateLimitSignals[keyID] = now
		return false
	}
	p.rateLimitSignals[keyID] = now
	if len(p.rateLimitSignals) < 2 {
		return false
	}
	if !p.lastConcurrencyDecrease.IsZero() && now.Sub(p.lastConcurrencyDecrease) < window {
		return false
	}

	currentLimit := p.effectiveMaxConcurrentLocked(now)
	if currentLimit <= 1 {
		return false
	}
	// Ceil(current/2) avoids dropping a seven-line pool below four slots on
	// the first confirmed account-wide signal.
	p.concurrencyBackoffLimit = (currentLimit + 1) / 2
	p.concurrencyRecoveryEvery = duration
	p.concurrencyBackoffUntil = now.Add(duration)
	p.lastConcurrencyDecrease = now
	p.cond.Broadcast()
	return true
}

func (p *Pool) calculateMinWaitDurationLocked(now time.Time) time.Duration {
	cutoff := now.Add(-60 * time.Second)
	var minExpiry time.Time

	for _, id := range p.keyOrder {
		item, ok := p.keys[id]
		if !ok || !item.Config.Enabled {
			continue
		}
		if item.CooldownUntil.After(now) {
			if minExpiry.IsZero() || item.CooldownUntil.Before(minExpiry) {
				minExpiry = item.CooldownUntil
			}
			continue
		}

		item.cleanTimestamps(cutoff)
		if len(item.Timestamps) >= item.Config.RPMLimit && len(item.Timestamps) > 0 {
			// The oldest timestamp will expire 60s after it was recorded
			expiresAt := item.Timestamps[0].Add(60 * time.Second)
			if minExpiry.IsZero() || expiresAt.Before(minExpiry) {
				minExpiry = expiresAt
			}
		}
	}

	if minExpiry.IsZero() {
		return 500 * time.Millisecond
	}

	diff := minExpiry.Sub(now)
	if diff < 10*time.Millisecond {
		return 10 * time.Millisecond
	}
	return diff
}

func (p *Pool) MarkKeyCooldown(keyID string, duration time.Duration) {
	p.mu.Lock()
	defer p.mu.Unlock()

	if item, ok := p.keys[keyID]; ok {
		until := time.Now().Add(duration)
		// A later short transport failure must not erase a longer Retry-After
		// already issued for this key.
		if until.After(item.CooldownUntil) {
			item.CooldownUntil = until
			p.cond.Broadcast()
		}
	}
}

// MarkKeyHeaderTimeout benches a key that produced no response headers within
// its entire first-byte budget. A full 60-150s window with zero bytes is strong
// evidence the key is not serving traffic (for example an exhausted free-tier
// line), so the cooldown escalates with consecutive occurrences instead of
// letting the key re-enter rotation after seconds. Any successful request
// resets the counter, so a recovered key is probed again at the short rung.
func (p *Pool) MarkKeyHeaderTimeout(keyID string) {
	p.mu.Lock()
	item, ok := p.keys[keyID]
	if !ok {
		p.mu.Unlock()
		return
	}
	item.ConsecutiveHeaderTimeouts++
	bench := 5 * time.Minute
	switch {
	case item.ConsecutiveHeaderTimeouts >= 3:
		bench = 30 * time.Minute
	case item.ConsecutiveHeaderTimeouts == 2:
		bench = 15 * time.Minute
	}
	until := time.Now().Add(bench)
	if until.After(item.CooldownUntil) {
		item.CooldownUntil = until
		p.cond.Broadcast()
	}
	p.mu.Unlock()
	// Benches are part of the durability contract: persist them so a restart
	// keeps avoiding this key instead of rediscovering it the hard way.
	_ = p.saveMetrics()
}

// AbortKey releases a reservation without recording success or failure.
// This is used when the client disconnects before the upstream responds; a
// caller-side cancellation is not evidence that the selected key is unhealthy.
func (p *Pool) AbortKey(keyID string) {
	p.mu.Lock()
	defer p.mu.Unlock()

	if item, ok := p.keys[keyID]; ok {
		if item.InFlightRequests > 0 {
			item.InFlightRequests--
		}
	}
	if p.activeRequests > 0 {
		p.activeRequests--
	}
	p.cond.Broadcast()
}

// EnabledKeyCount reports how many keys are currently enabled. The proxy uses
// it to bound per-request retries: every key may be tried about once before a
// request is declared unrecoverable.
func (p *Pool) EnabledKeyCount() int {
	p.mu.RLock()
	defer p.mu.RUnlock()
	count := 0
	for _, item := range p.keys {
		if item.Config.Enabled {
			count++
		}
	}
	return count
}

func (p *Pool) GetStats() PoolStats {
	// GetStats expires sliding-window entries and advances AIMD recovery, both
	// of which mutate pool state. Use the write lock so concurrent dashboard
	// polls cannot race on timestamp slice headers.
	p.mu.Lock()
	defer p.mu.Unlock()

	now := time.Now()
	cutoff := now.Add(-60 * time.Second)
	p.advanceConcurrencyRecoveryLocked(now)
	modelMappings := make(map[string]string, len(p.modelMappings))
	for alias, target := range p.modelMappings {
		modelMappings[alias] = target
	}

	stats := PoolStats{
		UpstreamURL:              p.upstreamURL,
		FreebuffBaseURL:          p.freebuffBaseURL,
		FreebuffEnabled:          p.freebuffEnabled,
		Port:                     p.port,
		TotalKeys:                len(p.keyOrder),
		QueuedRequests:           p.queuedCount,
		ActiveRequests:           p.activeRequests,
		MaxConcurrentRequests:    p.maxConcurrentRequests,
		EffectiveMaxConcurrent:   p.effectiveMaxConcurrentLocked(now),
		StripThinking:            p.stripThinking,
		FastStreamThinkingAsText: p.fastStreamThinkingAsText,
		DefaultEffort:            p.defaultEffort,
		SSEPingIntervalMs:        p.ssePingIntervalMs,
		ModelMappings:            modelMappings,
		AllowedModels:            p.getAllowedModelsLocked(),
		Keys:                     make([]KeyStatusDTO, 0, len(p.keyOrder)),
	}
	if p.concurrencyBackoffUntil.After(now) {
		stats.ConcurrencyBackoffSec = int(p.concurrencyBackoffUntil.Sub(now).Seconds()) + 1
	}

	var totalReq int64
	var aggMaxRPM int
	var aggCurrRPM int
	var activeCount int
	var eligibleCount int
	var busyCount int
	var cooldownCount int
	var rpmExhaustedCount int

	for _, id := range p.keyOrder {
		item, ok := p.keys[id]
		if !ok {
			continue
		}

		item.cleanTimestamps(cutoff)
		currRPM := len(item.Timestamps)
		totalReq += item.TotalRequests

		if item.Config.Enabled {
			activeCount++
			aggMaxRPM += item.Config.RPMLimit
			aggCurrRPM += currRPM
		}

		inCooldown := item.CooldownUntil.After(now)
		busy := item.InFlightRequests > 0
		rpmExhausted := item.Config.RPMLimit <= 0 || currRPM >= item.Config.RPMLimit
		eligible := item.Config.Enabled && !inCooldown && !rpmExhausted
		if busy {
			busyCount++
		}
		if item.Config.Enabled {
			if eligible {
				eligibleCount++
			}
			if inCooldown {
				cooldownCount++
			}
			if rpmExhausted {
				rpmExhaustedCount++
			}
		}
		cooldownRem := 0
		if inCooldown {
			cooldownRem = int(item.CooldownUntil.Sub(now).Seconds()) + 1
		}

		nextWinSec := 0
		if currRPM >= item.Config.RPMLimit && len(item.Timestamps) > 0 {
			exp := item.Timestamps[0].Add(60 * time.Second)
			if exp.After(now) {
				nextWinSec = int(exp.Sub(now).Seconds()) + 1
			}
		}

		masked := item.Config.Key
		if len(masked) > 12 {
			masked = masked[:7] + "..." + masked[len(masked)-4:]
		}

		stats.Keys = append(stats.Keys, KeyStatusDTO{
			ID:                   item.Config.ID,
			Name:                 item.Config.Name,
			KeyMasked:            masked,
			RPMLimit:             item.Config.RPMLimit,
			RPMCurrent:           currRPM,
			Enabled:              item.Config.Enabled,
			TotalRequests:        item.TotalRequests,
			SuccessCount:         item.SuccessCount,
			ErrorCount:           item.ErrorCount,
			InFlightRequests:     item.InFlightRequests,
			LastUsedTime:         item.LastUsedTime,
			InCooldown:           inCooldown,
			Eligible:             eligible,
			Busy:                 busy,
			RPMExhausted:         rpmExhausted,
			CooldownRemainingSec: cooldownRem,
			NextWindowSec:        nextWinSec,
			LatencyEMAMs:         item.LatencyEMAMs,
			LatencySamples:       item.LatencySamples,
			GapEMAMs:             item.GapEMAMs,
			GapSamples:           item.GapSamples,
		})
	}

	stats.ActiveKeys = activeCount
	stats.EligibleKeys = eligibleCount
	stats.BusyKeys = busyCount
	stats.CooldownKeys = cooldownCount
	stats.RPMExhaustedKeys = rpmExhaustedCount
	stats.AggregateMaxRPM = aggMaxRPM
	stats.AggregateCurrRPM = aggCurrRPM
	stats.TotalRequests = totalReq
	if stats.QueuedRequests > 0 {
		switch {
		case stats.EffectiveMaxConcurrent > 0 && stats.ActiveRequests >= stats.EffectiveMaxConcurrent:
			stats.QueueBlockReason = "concurrency_saturated"
		case stats.EligibleKeys == 0 && stats.CooldownKeys == stats.ActiveKeys && stats.ActiveKeys > 0:
			stats.QueueBlockReason = "cooldown"
		case stats.EligibleKeys == 0 && stats.RPMExhaustedKeys == stats.ActiveKeys && stats.ActiveKeys > 0:
			stats.QueueBlockReason = "rpm_exhausted"
		default:
			stats.QueueBlockReason = "waiting_for_dispatch"
		}
	}

	return stats
}

// RecordKeyLatency updates a small per-key EWMA from successful upstream
// response-header latency. It lets selection favor a faster rotated path while
// still honoring RPM and the global concurrency ceiling.
func (p *Pool) RecordKeyLatency(keyID string, latencyMs int64) {
	if latencyMs <= 0 {
		return
	}
	p.mu.Lock()
	item, ok := p.keys[keyID]
	if !ok {
		p.mu.Unlock()
		return
	}
	if item.LatencySamples == 0 {
		item.LatencyEMAMs = latencyMs
	} else {
		// React quickly when TokenRouter moves a key between different provider
		// workers. A 50/50 EMA keeps the picker from clinging to a path that was
		// fast several requests ago but is now queued.
		item.LatencyEMAMs = (item.LatencyEMAMs + latencyMs) / 2
	}
	item.LatencySamples++
	p.mu.Unlock()
	_ = p.saveMetrics()
}

// RecordKeyStreamQuality tracks the largest inter-chunk gap on a successful
// stream. The selector uses it as a secondary smoothness signal.
func (p *Pool) RecordKeyStreamQuality(keyID string, maxGapMs int64) {
	if maxGapMs <= 0 {
		return
	}
	p.mu.Lock()
	item, ok := p.keys[keyID]
	if !ok {
		p.mu.Unlock()
		return
	}
	if item.GapSamples == 0 {
		item.GapEMAMs = maxGapMs
	} else {
		// Gap spikes are immediately visible to the user, so react faster than
		// the first-byte EMA and let a recent stall move a key out of the hot path.
		item.GapEMAMs = (item.GapEMAMs + maxGapMs) / 2
	}
	item.GapSamples++
	p.mu.Unlock()
	_ = p.saveMetrics()
}

// Management API handlers
func (p *Pool) SetUpstreamURL(urlStr string) error {
	p.mu.Lock()
	p.upstreamURL = urlStr
	p.mu.Unlock()
	return p.SaveConfig()
}

func (p *Pool) ToggleKey(keyID string, enabled bool) error {
	p.mu.Lock()
	if item, ok := p.keys[keyID]; ok {
		item.Config.Enabled = enabled
	}
	p.cond.Broadcast()
	p.mu.Unlock()
	return p.SaveConfig()
}

func (p *Pool) UpdateKey(cfg KeyConfig) error {
	p.mu.Lock()
	if item, ok := p.keys[cfg.ID]; ok {
		item.Config.Name = cfg.Name
		item.Config.Key = cfg.Key
		item.Config.RPMLimit = cfg.RPMLimit
		item.Config.Enabled = cfg.Enabled
	} else {
		p.keyOrder = append(p.keyOrder, cfg.ID)
		p.keys[cfg.ID] = &KeyItem{
			Config:     cfg,
			Timestamps: make([]time.Time, 0),
		}
	}
	p.cond.Broadcast()
	p.mu.Unlock()
	return p.SaveConfig()
}

func (p *Pool) DeleteKey(keyID string) error {
	p.mu.Lock()
	delete(p.keys, keyID)
	newOrder := make([]string, 0, len(p.keyOrder))
	for _, id := range p.keyOrder {
		if id != keyID {
			newOrder = append(newOrder, id)
		}
	}
	p.keyOrder = newOrder
	p.cond.Broadcast()
	p.mu.Unlock()
	return p.SaveConfig()
}

func (p *Pool) GetUpstreamURL() string {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.upstreamURL
}

func (p *Pool) GetFreebuffBaseURL() string {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.freebuffBaseURL
}

func (p *Pool) IsFreebuffEnabled() bool {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.freebuffEnabled
}

func (p *Pool) SetFreebuffBaseURL(urlStr string) error {
	p.mu.Lock()
	p.freebuffBaseURL = urlStr
	p.mu.Unlock()
	return p.SaveConfig()
}

func (p *Pool) SetFreebuffEnabled(enabled bool) error {
	p.mu.Lock()
	p.freebuffEnabled = enabled
	p.mu.Unlock()
	return p.SaveConfig()
}

func (p *Pool) GetPort() int {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.port
}

// GetAllowedModels returns the configured model allowlist. An empty list means
// backwards-compatible unrestricted routing.
func (p *Pool) GetAllowedModels() []string {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.getAllowedModelsLocked()
}

func (p *Pool) getAllowedModelsLocked() []string {
	models := make([]string, 0, len(p.allowedModels))
	for model := range p.allowedModels {
		models = append(models, model)
	}
	sort.Strings(models)
	return models
}

func (p *Pool) IsModelAllowed(model string) bool {
	p.mu.RLock()
	defer p.mu.RUnlock()

	if len(p.allowedModels) == 0 {
		return true
	}
	_, ok := p.allowedModels[strings.TrimSpace(model)]
	return ok
}

func (p *Pool) ResolveModelAlias(requestedModel string) string {
	p.mu.RLock()
	defer p.mu.RUnlock()

	if target, exists := p.modelMappings[requestedModel]; exists && target != "" {
		return target
	}
	// Claude clients keep minting dated model IDs (haiku refreshes, new sonnet
	// versions) and a missing alias passes the raw name upstream, which
	// rejects the unknown model with a fast 503 on every key. Any unmapped
	// claude-* request is still meant for the mapped Claude line, so fall
	// back to the configured sonnet target instead of forwarding the unknown
	// name.
	if strings.HasPrefix(requestedModel, "claude-") {
		for _, alias := range []string{"claude-3-5-sonnet", "claude-sonnet-4-5", "claude-sonnet-4"} {
			if target, exists := p.modelMappings[alias]; exists && target != "" {
				return target
			}
		}
		for alias, target := range p.modelMappings {
			if strings.HasPrefix(alias, "claude-") && target != "" {
				return target
			}
		}
	}
	return requestedModel
}

func (p *Pool) GetStripThinking() bool {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.stripThinking
}

func (p *Pool) SetStripThinking(strip bool) error {
	p.mu.Lock()
	p.stripThinking = strip
	p.mu.Unlock()
	return p.SaveConfig()
}

func (p *Pool) GetSSEPingIntervalMs() int {
	p.mu.RLock()
	defer p.mu.RUnlock()
	if p.ssePingIntervalMs <= 0 {
		return 500
	}
	return p.ssePingIntervalMs
}

func (p *Pool) SetModelMapping(alias, target string) error {
	p.mu.Lock()
	if p.modelMappings == nil {
		p.modelMappings = make(map[string]string)
	}
	if target == "" {
		delete(p.modelMappings, alias)
	} else {
		p.modelMappings[alias] = target
	}
	p.mu.Unlock()
	return p.SaveConfig()
}

func (p *Pool) GetFastStreamThinkingAsText() bool {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.fastStreamThinkingAsText
}

func (p *Pool) SetFastStreamThinkingAsText(enable bool) error {
	p.mu.Lock()
	p.fastStreamThinkingAsText = enable
	p.mu.Unlock()
	return p.SaveConfig()
}

func (p *Pool) GetDefaultEffort() string {
	p.mu.RLock()
	defer p.mu.RUnlock()
	if p.defaultEffort == "" {
		return "low"
	}
	return p.defaultEffort
}

func (p *Pool) SetDefaultEffort(effort string) error {
	p.mu.Lock()
	p.defaultEffort = effort
	p.mu.Unlock()
	return p.SaveConfig()
}
