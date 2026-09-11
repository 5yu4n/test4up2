package router

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestPoolHonorsGlobalConcurrentCeiling(t *testing.T) {
	config := `{"port":8080,"upstream_url":"http://127.0.0.1:1","max_queue_seconds":2,"max_concurrent_requests":1,"keys":[{"id":"k1","name":"K1","key":"test-key","rpm_limit":10,"enabled":true}]}`
	path := filepath.Join(t.TempDir(), "config.json")
	if err := os.WriteFile(path, []byte(config), 0600); err != nil {
		t.Fatal(err)
	}
	p, err := NewPool(path)
	if err != nil {
		t.Fatal(err)
	}

	_, releaseFirst, err := p.AcquireKey(context.Background(), nil)
	if err != nil {
		t.Fatal(err)
	}

	secondDone := make(chan struct{})
	go func() {
		_, releaseSecond, acquireErr := p.AcquireKey(context.Background(), nil)
		if acquireErr == nil {
			releaseSecond(true)
		}
		close(secondDone)
	}()

	select {
	case <-secondDone:
		t.Fatal("second request bypassed the concurrent ceiling")
	case <-time.After(150 * time.Millisecond):
	}
	if stats := p.GetStats(); stats.QueuedRequests != 1 || stats.QueueBlockReason != "concurrency_saturated" {
		t.Fatalf("queued capacity state = queued:%d reason:%q, want 1/concurrency_saturated", stats.QueuedRequests, stats.QueueBlockReason)
	}

	releaseFirst(true)
	select {
	case <-secondDone:
	case <-time.After(1 * time.Second):
		t.Fatal("queued request did not proceed after the first request released")
	}
}

func TestPoolPrefersLowerObservedLatencyWhenLoadIsTied(t *testing.T) {
	config := `{"port":8080,"upstream_url":"http://127.0.0.1:1","max_queue_seconds":1,"keys":[{"id":"slow","name":"Slow","key":"slow","rpm_limit":10,"enabled":true},{"id":"fast","name":"Fast","key":"fast","rpm_limit":10,"enabled":true}]}`
	path := filepath.Join(t.TempDir(), "config.json")
	if err := os.WriteFile(path, []byte(config), 0600); err != nil {
		t.Fatal(err)
	}
	p, err := NewPool(path)
	if err != nil {
		t.Fatal(err)
	}
	p.RecordKeyLatency("slow", 5000)
	p.RecordKeyLatency("fast", 500)

	key, release, err := p.AcquireKey(context.Background(), nil)
	if err != nil {
		t.Fatal(err)
	}
	defer release(true)
	if key.Config.ID != "fast" {
		t.Fatalf("selected key = %q, want fast key", key.Config.ID)
	}
}

func TestPoolSpreadsConcurrentWorkAcrossKeys(t *testing.T) {
	config := `{"port":8080,"upstream_url":"http://127.0.0.1:1","max_queue_seconds":1,"keys":[{"id":"slow","name":"Slow","key":"slow","rpm_limit":10,"enabled":true},{"id":"fast","name":"Fast","key":"fast","rpm_limit":10,"enabled":true}]}`
	path := filepath.Join(t.TempDir(), "config.json")
	if err := os.WriteFile(path, []byte(config), 0600); err != nil {
		t.Fatal(err)
	}
	p, err := NewPool(path)
	if err != nil {
		t.Fatal(err)
	}
	p.RecordKeyLatency("slow", 10000)
	p.RecordKeyLatency("fast", 500)
	p.keys["fast"].InFlightRequests = 1
	p.activeRequests = 1

	key, release, err := p.AcquireKey(context.Background(), nil)
	if err != nil {
		t.Fatal(err)
	}
	defer release(true)
	if key.Config.ID != "slow" {
		t.Fatalf("selected key = %q, want idle key so concurrent work is spread", key.Config.ID)
	}
}

func TestPoolSpreadsRPMReservationsBeforeReusingAKey(t *testing.T) {
	config := `{"port":8080,"upstream_url":"http://127.0.0.1:1","max_queue_seconds":1,"max_concurrent_requests":7,"keys":[{"id":"fast","name":"Fast","key":"fast","rpm_limit":10,"enabled":true},{"id":"slow","name":"Slow","key":"slow","rpm_limit":10,"enabled":true}]}`
	path := filepath.Join(t.TempDir(), "config.json")
	if err := os.WriteFile(path, []byte(config), 0600); err != nil {
		t.Fatal(err)
	}
	p, err := NewPool(path)
	if err != nil {
		t.Fatal(err)
	}
	p.RecordKeyLatency("fast", 100)
	p.RecordKeyLatency("slow", 10000)

	first, releaseFirst, err := p.AcquireKey(context.Background(), nil)
	if err != nil {
		t.Fatal(err)
	}
	releaseFirst(true)
	second, releaseSecond, err := p.AcquireKey(context.Background(), nil)
	if err != nil {
		t.Fatal(err)
	}
	defer releaseSecond(true)
	if first.Config.ID != "fast" || second.Config.ID != "slow" {
		t.Fatalf("selection order = %q, %q; want fast then unused slow key", first.Config.ID, second.Config.ID)
	}
}

func TestPoolFillsSevenDistinctLinesBeforeQueueingEighth(t *testing.T) {
	config := `{"port":8080,"upstream_url":"http://127.0.0.1:1","max_queue_seconds":2,"max_concurrent_requests":7,"keys":[` +
		`{"id":"k1","name":"K1","key":"one","rpm_limit":15,"enabled":true},` +
		`{"id":"k2","name":"K2","key":"two","rpm_limit":15,"enabled":true},` +
		`{"id":"k3","name":"K3","key":"three","rpm_limit":15,"enabled":true},` +
		`{"id":"k4","name":"K4","key":"four","rpm_limit":15,"enabled":true},` +
		`{"id":"k5","name":"K5","key":"five","rpm_limit":15,"enabled":true},` +
		`{"id":"k6","name":"K6","key":"six","rpm_limit":15,"enabled":true},` +
		`{"id":"k7","name":"K7","key":"seven","rpm_limit":15,"enabled":true}]}`
	path := filepath.Join(t.TempDir(), "config.json")
	if err := os.WriteFile(path, []byte(config), 0600); err != nil {
		t.Fatal(err)
	}
	p, err := NewPool(path)
	if err != nil {
		t.Fatal(err)
	}

	releases := make([]func(bool), 0, 7)
	selected := make(map[string]bool, 7)
	for i := 0; i < 7; i++ {
		key, release, acquireErr := p.AcquireKey(context.Background(), nil)
		if acquireErr != nil {
			t.Fatalf("acquire line %d: %v", i+1, acquireErr)
		}
		if selected[key.Config.ID] {
			t.Fatalf("key %s reused before all seven lines were occupied", key.Config.ID)
		}
		selected[key.Config.ID] = true
		releases = append(releases, release)
	}
	if got := p.GetStats().ActiveRequests; got != 7 {
		t.Fatalf("active requests = %d, want all seven lines occupied", got)
	}

	eighthAcquired := make(chan string, 1)
	go func() {
		key, release, acquireErr := p.AcquireKey(context.Background(), nil)
		if acquireErr != nil {
			eighthAcquired <- "error: " + acquireErr.Error()
			return
		}
		release(true)
		eighthAcquired <- key.Config.ID
	}()
	select {
	case result := <-eighthAcquired:
		t.Fatalf("eighth request bypassed full seven-line ceiling: %s", result)
	case <-time.After(100 * time.Millisecond):
	}

	releases[0](true)
	select {
	case result := <-eighthAcquired:
		if result != "k1" {
			t.Fatalf("eighth request used %s after k1 released, want k1", result)
		}
	case <-time.After(1 * time.Second):
		t.Fatal("eighth request did not enter promptly after a line released")
	}
	for _, release := range releases[1:] {
		release(true)
	}
	if got := p.GetStats().ActiveRequests; got != 0 {
		t.Fatalf("active requests after releases = %d, want 0", got)
	}
}

func TestPoolRecordsStreamGapAndExposesItInStats(t *testing.T) {
	config := `{"port":8080,"upstream_url":"http://127.0.0.1:1","max_queue_seconds":1,"keys":[{"id":"k1","name":"K1","key":"test-key","rpm_limit":10,"enabled":true}]}`
	path := filepath.Join(t.TempDir(), "config.json")
	if err := os.WriteFile(path, []byte(config), 0600); err != nil {
		t.Fatal(err)
	}
	p, err := NewPool(path)
	if err != nil {
		t.Fatal(err)
	}
	p.RecordKeyStreamQuality("k1", 1200)
	p.RecordKeyStreamQuality("k1", 400)

	stats := p.GetStats()
	if got := stats.Keys[0].GapEMAMs; got != 800 {
		t.Fatalf("gap EMA = %d, want 800", got)
	}
	if stats.Keys[0].GapSamples != 2 {
		t.Fatalf("gap samples = %d, want 2", stats.Keys[0].GapSamples)
	}
}

func TestPoolRestoresLatencyMetricsAcrossReload(t *testing.T) {
	config := `{"port":8080,"upstream_url":"http://127.0.0.1:1","max_queue_seconds":1,"keys":[{"id":"k1","name":"K1","key":"test-key","rpm_limit":10,"enabled":true}]}`
	path := filepath.Join(t.TempDir(), "config.json")
	if err := os.WriteFile(path, []byte(config), 0600); err != nil {
		t.Fatal(err)
	}
	p, err := NewPool(path)
	if err != nil {
		t.Fatal(err)
	}
	p.RecordKeyLatency("k1", 1234)
	p.RecordKeyStreamQuality("k1", 456)
	// Legacy sidecars may contain error counters; those are intentionally
	// ignored across restarts so a transient throttle cannot starve a key.
	legacyMetrics := `{"k1":{"latency_ema_ms":1234,"latency_samples":1,"gap_ema_ms":456,"gap_samples":1,"consecutive_errors":3}}`
	if err := os.WriteFile(path+".metrics.json", []byte(legacyMetrics), 0600); err != nil {
		t.Fatal(err)
	}

	reloaded, err := NewPool(path)
	if err != nil {
		t.Fatal(err)
	}
	stats := reloaded.GetStats()
	if got := stats.Keys[0].LatencyEMAMs; got != 1234 {
		t.Fatalf("restored latency EMA = %d, want 1234", got)
	}
	if got := stats.Keys[0].GapEMAMs; got != 456 {
		t.Fatalf("restored gap EMA = %d, want 456", got)
	}
	if got := reloaded.keys["k1"].ConsecutiveErrors; got != 0 {
		t.Fatalf("restored consecutive errors = %d, want 0", got)
	}
}

func TestPoolRequiresDistinctRateLimitedKeysBeforeAIMD(t *testing.T) {
	config := `{"port":8080,"upstream_url":"http://127.0.0.1:1","max_queue_seconds":1,"max_concurrent_requests":4,"keys":[` +
		`{"id":"k1","name":"K1","key":"one","rpm_limit":10,"enabled":true},` +
		`{"id":"k2","name":"K2","key":"two","rpm_limit":10,"enabled":true},` +
		`{"id":"k3","name":"K3","key":"three","rpm_limit":10,"enabled":true},` +
		`{"id":"k4","name":"K4","key":"four","rpm_limit":10,"enabled":true}]}`
	path := filepath.Join(t.TempDir(), "config.json")
	if err := os.WriteFile(path, []byte(config), 0600); err != nil {
		t.Fatal(err)
	}
	p, err := NewPool(path)
	if err != nil {
		t.Fatal(err)
	}

	if got := p.GetStats().EffectiveMaxConcurrent; got != 4 {
		t.Fatalf("initial effective concurrency = %d, want 4", got)
	}
	if lowered := p.RecordUpstreamRateLimit("k1", 5*time.Second); lowered {
		t.Fatal("one key-specific throttle lowered account concurrency")
	}
	if lowered := p.RecordUpstreamRateLimit("k1", 5*time.Second); lowered {
		t.Fatal("a repeated signal from one key lowered account concurrency")
	}
	if got := p.GetStats().EffectiveMaxConcurrent; got != 4 {
		t.Fatalf("effective concurrency after one distinct key = %d, want 4", got)
	}

	if lowered := p.RecordUpstreamRateLimit("k2", 5*time.Second); !lowered {
		t.Fatal("two distinct throttled keys did not trigger multiplicative decrease")
	}
	stats := p.GetStats()
	if got := stats.EffectiveMaxConcurrent; got != 2 {
		t.Fatalf("effective concurrency after distinct-key quorum = %d, want 2", got)
	}
	if stats.ConcurrencyBackoffSec < 4 {
		t.Fatalf("recovery wait = %d, want at least 4s", stats.ConcurrencyBackoffSec)
	}
	if lowered := p.RecordUpstreamRateLimit("k3", 5*time.Second); lowered {
		t.Fatal("one signal wave lowered concurrency more than once")
	}
	if got := p.GetStats().EffectiveMaxConcurrent; got != 2 {
		t.Fatalf("effective concurrency after same-wave third key = %d, want 2", got)
	}
}

func TestPoolAIMDRecoversOneSlotPerInterval(t *testing.T) {
	config := `{"port":8080,"upstream_url":"http://127.0.0.1:1","max_queue_seconds":1,"max_concurrent_requests":4,"keys":[` +
		`{"id":"k1","name":"K1","key":"one","rpm_limit":10,"enabled":true},` +
		`{"id":"k2","name":"K2","key":"two","rpm_limit":10,"enabled":true}]}`
	path := filepath.Join(t.TempDir(), "config.json")
	if err := os.WriteFile(path, []byte(config), 0600); err != nil {
		t.Fatal(err)
	}
	p, err := NewPool(path)
	if err != nil {
		t.Fatal(err)
	}
	p.RecordUpstreamRateLimit("k1", 5*time.Second)
	p.RecordUpstreamRateLimit("k2", 5*time.Second)
	if got := p.GetStats().EffectiveMaxConcurrent; got != 2 {
		t.Fatalf("decreased concurrency = %d, want 2", got)
	}

	p.mu.Lock()
	p.concurrencyBackoffUntil = time.Now().Add(-time.Millisecond)
	p.mu.Unlock()
	if got := p.GetStats().EffectiveMaxConcurrent; got != 3 {
		t.Fatalf("first additive recovery = %d, want 3", got)
	}

	p.mu.Lock()
	p.concurrencyBackoffUntil = time.Now().Add(-time.Millisecond)
	p.mu.Unlock()
	if got := p.GetStats().EffectiveMaxConcurrent; got != 4 {
		t.Fatalf("second additive recovery = %d, want configured ceiling 4", got)
	}
}

func TestPoolSingleRateLimitedKeyDoesNotBlockHealthyLine(t *testing.T) {
	config := `{"port":8080,"upstream_url":"http://127.0.0.1:1","max_queue_seconds":1,"max_concurrent_requests":2,"keys":[` +
		`{"id":"k1","name":"K1","key":"one","rpm_limit":10,"enabled":true},` +
		`{"id":"k2","name":"K2","key":"two","rpm_limit":10,"enabled":true}]}`
	path := filepath.Join(t.TempDir(), "config.json")
	if err := os.WriteFile(path, []byte(config), 0600); err != nil {
		t.Fatal(err)
	}
	p, err := NewPool(path)
	if err != nil {
		t.Fatal(err)
	}
	p.MarkKeyCooldown("k1", time.Second)
	p.RecordUpstreamRateLimit("k1", time.Second)

	key, release, err := p.AcquireKey(context.Background(), nil)
	if err != nil {
		t.Fatal(err)
	}
	defer release(true)
	if key.Config.ID != "k2" {
		t.Fatalf("selected key = %q, want healthy k2", key.Config.ID)
	}
	if got := p.GetStats().EffectiveMaxConcurrent; got != 2 {
		t.Fatalf("single-key throttle changed global ceiling to %d", got)
	}
}

func TestPoolCooldownNeverShrinks(t *testing.T) {
	config := `{"port":8080,"upstream_url":"http://127.0.0.1:1","max_queue_seconds":1,"keys":[{"id":"k1","name":"K1","key":"one","rpm_limit":10,"enabled":true}]}`
	path := filepath.Join(t.TempDir(), "config.json")
	if err := os.WriteFile(path, []byte(config), 0600); err != nil {
		t.Fatal(err)
	}
	p, err := NewPool(path)
	if err != nil {
		t.Fatal(err)
	}
	p.MarkKeyCooldown("k1", 5*time.Second)
	first := p.keys["k1"].CooldownUntil
	p.MarkKeyCooldown("k1", 100*time.Millisecond)
	if got := p.keys["k1"].CooldownUntil; !got.Equal(first) {
		t.Fatalf("short cooldown replaced longer deadline: first=%s got=%s", first, got)
	}
	p.MarkKeyCooldown("k1", 6*time.Second)
	if got := p.keys["k1"].CooldownUntil; !got.After(first) {
		t.Fatalf("longer cooldown did not extend deadline: first=%s got=%s", first, got)
	}
}

func TestPoolCanceledWaiterWakesWithoutSlotRelease(t *testing.T) {
	config := `{"port":8080,"upstream_url":"http://127.0.0.1:1","max_queue_seconds":2,"max_concurrent_requests":1,"keys":[{"id":"k1","name":"K1","key":"one","rpm_limit":10,"enabled":true}]}`
	path := filepath.Join(t.TempDir(), "config.json")
	if err := os.WriteFile(path, []byte(config), 0600); err != nil {
		t.Fatal(err)
	}
	p, err := NewPool(path)
	if err != nil {
		t.Fatal(err)
	}
	_, release, err := p.AcquireKey(context.Background(), nil)
	if err != nil {
		t.Fatal(err)
	}
	defer release(true)

	ctx, cancel := context.WithCancel(context.Background())
	waiterDone := make(chan error, 1)
	go func() {
		_, _, acquireErr := p.AcquireKey(ctx, nil)
		waiterDone <- acquireErr
	}()
	deadline := time.Now().Add(time.Second)
	for p.GetStats().QueuedRequests == 0 && time.Now().Before(deadline) {
		time.Sleep(time.Millisecond)
	}
	cancel()
	select {
	case acquireErr := <-waiterDone:
		if acquireErr == nil || !errors.Is(acquireErr, context.Canceled) {
			t.Fatalf("canceled waiter error = %v, want context.Canceled", acquireErr)
		}
	case <-time.After(250 * time.Millisecond):
		t.Fatal("canceled waiter missed its condition wakeup")
	}
}

func TestPoolStatsClassifyCapacityAndCloneMappings(t *testing.T) {
	config := `{"port":8080,"upstream_url":"http://127.0.0.1:1","max_queue_seconds":1,"max_concurrent_requests":4,"model_mappings":{"alias":"target"},"keys":[` +
		`{"id":"idle","name":"Idle","key":"one","rpm_limit":1,"enabled":true},` +
		`{"id":"busy","name":"Busy","key":"two","rpm_limit":1,"enabled":true},` +
		`{"id":"cool","name":"Cool","key":"three","rpm_limit":1,"enabled":true},` +
		`{"id":"rpm","name":"RPM","key":"four","rpm_limit":1,"enabled":true},` +
		`{"id":"off","name":"Off","key":"five","rpm_limit":1,"enabled":false}]}`
	path := filepath.Join(t.TempDir(), "config.json")
	if err := os.WriteFile(path, []byte(config), 0600); err != nil {
		t.Fatal(err)
	}
	p, err := NewPool(path)
	if err != nil {
		t.Fatal(err)
	}
	p.mu.Lock()
	p.keys["busy"].InFlightRequests = 1
	p.activeRequests = 1
	p.keys["cool"].CooldownUntil = time.Now().Add(time.Minute)
	p.keys["rpm"].Timestamps = append(p.keys["rpm"].Timestamps, time.Now())
	p.queuedCount = 1
	p.mu.Unlock()

	stats := p.GetStats()
	if stats.ActiveKeys != 4 || stats.EligibleKeys != 2 || stats.BusyKeys != 1 || stats.CooldownKeys != 1 || stats.RPMExhaustedKeys != 1 {
		t.Fatalf("capacity classes = active:%d eligible:%d busy:%d cooldown:%d rpm:%d", stats.ActiveKeys, stats.EligibleKeys, stats.BusyKeys, stats.CooldownKeys, stats.RPMExhaustedKeys)
	}
	if stats.QueueBlockReason != "waiting_for_dispatch" {
		t.Fatalf("mixed capacity queue reason = %q, want waiting_for_dispatch", stats.QueueBlockReason)
	}
	byID := make(map[string]KeyStatusDTO, len(stats.Keys))
	for _, key := range stats.Keys {
		byID[key.ID] = key
	}
	if !byID["idle"].Eligible || !byID["busy"].Eligible || !byID["busy"].Busy || !byID["cool"].InCooldown || !byID["rpm"].RPMExhausted || byID["off"].Eligible {
		t.Fatalf("unexpected per-key classifications: %#v", byID)
	}
	stats.ModelMappings["alias"] = "mutated"
	if got := p.ResolveModelAlias("alias"); got != "target" {
		t.Fatalf("stats exposed mutable model map: pool target = %q", got)
	}
}

func TestPoolStatsExplainCooldownAndRPMQueues(t *testing.T) {
	config := `{"port":8080,"upstream_url":"http://127.0.0.1:1","max_queue_seconds":1,"max_concurrent_requests":2,"keys":[` +
		`{"id":"k1","name":"K1","key":"one","rpm_limit":1,"enabled":true},` +
		`{"id":"k2","name":"K2","key":"two","rpm_limit":1,"enabled":true}]}`
	path := filepath.Join(t.TempDir(), "config.json")
	if err := os.WriteFile(path, []byte(config), 0600); err != nil {
		t.Fatal(err)
	}
	p, err := NewPool(path)
	if err != nil {
		t.Fatal(err)
	}

	p.mu.Lock()
	p.queuedCount = 1
	for _, item := range p.keys {
		item.CooldownUntil = time.Now().Add(time.Minute)
	}
	p.mu.Unlock()
	if got := p.GetStats().QueueBlockReason; got != "cooldown" {
		t.Fatalf("cooldown queue reason = %q", got)
	}

	p.mu.Lock()
	for _, item := range p.keys {
		item.CooldownUntil = time.Time{}
		item.Timestamps = []time.Time{time.Now()}
	}
	p.mu.Unlock()
	if got := p.GetStats().QueueBlockReason; got != "rpm_exhausted" {
		t.Fatalf("RPM queue reason = %q", got)
	}
}

func TestPoolQueueDeadlineCapsLongRPMWait(t *testing.T) {
	config := `{"port":8080,"upstream_url":"http://127.0.0.1:1","max_queue_seconds":1,"max_concurrent_requests":2,"keys":[{"id":"k1","name":"K1","key":"one","rpm_limit":1,"enabled":true}]}`
	path := filepath.Join(t.TempDir(), "config.json")
	if err := os.WriteFile(path, []byte(config), 0600); err != nil {
		t.Fatal(err)
	}
	p, err := NewPool(path)
	if err != nil {
		t.Fatal(err)
	}
	_, release, err := p.AcquireKey(context.Background(), nil)
	if err != nil {
		t.Fatal(err)
	}
	release(true)

	started := time.Now()
	_, _, err = p.AcquireKey(context.Background(), nil)
	if err == nil || !strings.Contains(err.Error(), "queue timeout") {
		t.Fatalf("RPM-saturated acquire error = %v, want queue timeout", err)
	}
	if elapsed := time.Since(started); elapsed > 1500*time.Millisecond {
		t.Fatalf("queue deadline was hidden behind RPM expiry: %s", elapsed)
	}
}

func TestPoolHeaderTimeoutEscalatesCooldown(t *testing.T) {
	config := `{"port":8080,"upstream_url":"http://127.0.0.1:1","max_queue_seconds":1,"keys":[{"id":"k1","name":"K1","key":"one","rpm_limit":10,"enabled":true}]}`
	path := filepath.Join(t.TempDir(), "config.json")
	if err := os.WriteFile(path, []byte(config), 0600); err != nil {
		t.Fatal(err)
	}
	p, err := NewPool(path)
	if err != nil {
		t.Fatal(err)
	}

	benchDuration := func() time.Duration {
		p.mu.RLock()
		defer p.mu.RUnlock()
		return time.Until(p.keys["k1"].CooldownUntil)
	}

	p.MarkKeyHeaderTimeout("k1")
	if got := p.keys["k1"].ConsecutiveHeaderTimeouts; got != 1 {
		t.Fatalf("consecutive header timeouts = %d, want 1", got)
	}
	if got := benchDuration(); got < 4*time.Minute || got > 6*time.Minute {
		t.Fatalf("first bench = %s, want ~5m", got)
	}

	p.MarkKeyHeaderTimeout("k1")
	if got := benchDuration(); got < 14*time.Minute || got > 16*time.Minute {
		t.Fatalf("second bench = %s, want ~15m", got)
	}

	p.MarkKeyHeaderTimeout("k1")
	if got := benchDuration(); got < 29*time.Minute || got > 31*time.Minute {
		t.Fatalf("third bench = %s, want ~30m", got)
	}
	p.MarkKeyHeaderTimeout("k1")
	if got := benchDuration(); got < 29*time.Minute {
		t.Fatalf("fourth bench = %s, want capped 30m", got)
	}

	// A transient short cooldown must not shorten an escalated bench.
	before := benchDuration()
	p.MarkKeyCooldown("k1", 10*time.Second)
	if got := benchDuration(); got < before {
		t.Fatalf("short cooldown shortened escalated bench: %s < %s", got, before)
	}
}

func TestPoolSuccessResetsHeaderTimeoutEscalation(t *testing.T) {
	config := `{"port":8080,"upstream_url":"http://127.0.0.1:1","max_queue_seconds":1,"keys":[{"id":"k1","name":"K1","key":"one","rpm_limit":10,"enabled":true}]}`
	path := filepath.Join(t.TempDir(), "config.json")
	if err := os.WriteFile(path, []byte(config), 0600); err != nil {
		t.Fatal(err)
	}
	p, err := NewPool(path)
	if err != nil {
		t.Fatal(err)
	}

	p.MarkKeyHeaderTimeout("k1")
	p.MarkKeyHeaderTimeout("k1")
	if got := p.keys["k1"].ConsecutiveHeaderTimeouts; got != 2 {
		t.Fatalf("consecutive header timeouts = %d, want 2", got)
	}

	// Clear the bench so the key can be acquired again, then complete a
	// successful request through the pool's own release path.
	p.mu.Lock()
	p.keys["k1"].CooldownUntil = time.Time{}
	p.mu.Unlock()
	_, release, err := p.AcquireKey(context.Background(), nil)
	if err != nil {
		t.Fatal(err)
	}
	release(true)
	if got := p.keys["k1"].ConsecutiveHeaderTimeouts; got != 0 {
		t.Fatalf("consecutive header timeouts after success = %d, want 0", got)
	}

	p.MarkKeyHeaderTimeout("k1")
	if got := p.keys["k1"].ConsecutiveHeaderTimeouts; got != 1 {
		t.Fatalf("escalation restarted at %d, want 1", got)
	}
	p.mu.RLock()
	bench := time.Until(p.keys["k1"].CooldownUntil)
	p.mu.RUnlock()
	if bench < 4*time.Minute || bench > 6*time.Minute {
		t.Fatalf("post-reset bench = %s, want ~5m", bench)
	}
}

func TestPoolResolvesUnmappedClaudeAliasToSonnetLine(t *testing.T) {
	config := `{"port":8080,"upstream_url":"http://127.0.0.1:1","max_queue_seconds":1,"model_mappings":{"claude-3-5-sonnet":"z-ai/glm-5.3-free","claude-3-5-sonnet-20241022":"z-ai/glm-5.3-free"},"keys":[{"id":"k1","name":"K1","key":"one","rpm_limit":10,"enabled":true}]}`
	path := filepath.Join(t.TempDir(), "config.json")
	if err := os.WriteFile(path, []byte(config), 0600); err != nil {
		t.Fatal(err)
	}
	p, err := NewPool(path)
	if err != nil {
		t.Fatal(err)
	}

	// Exact matches keep winning.
	if got := p.ResolveModelAlias("claude-3-5-sonnet-20241022"); got != "z-ai/glm-5.3-free" {
		t.Fatalf("exact alias = %q, want mapped target", got)
	}
	// An unmapped dated Claude ID (for example a mistyped config alias)
	// must fall back to the mapped Claude line instead of passing the raw
	// name upstream, where it is rejected with a fast 503 on every key.
	if got := p.ResolveModelAlias("claude-3-5-haiku-20240307"); got != "z-ai/glm-5.3-free" {
		t.Fatalf("unmapped claude alias = %q, want sonnet-line target", got)
	}
	if got := p.ResolveModelAlias("claude-sonnet-4-6-20990101"); got != "z-ai/glm-5.3-free" {
		t.Fatalf("future claude alias = %q, want sonnet-line target", got)
	}
	// Non-Claude models pass through untouched.
	if got := p.ResolveModelAlias("z-ai/glm-5.3-free"); got != "z-ai/glm-5.3-free" {
		t.Fatalf("passthrough model rewritten: %q", got)
	}
	if got := p.ResolveModelAlias("gpt-4o"); got != "gpt-4o" {
		t.Fatalf("unrelated model rewritten: %q", got)
	}
}

func TestPoolPersistsHeaderTimeoutBenchAcrossReload(t *testing.T) {
	config := `{"port":8080,"upstream_url":"http://127.0.0.1:1","max_queue_seconds":1,"keys":[{"id":"k1","name":"K1","key":"one","rpm_limit":10,"enabled":true}]}`
	path := filepath.Join(t.TempDir(), "config.json")
	if err := os.WriteFile(path, []byte(config), 0600); err != nil {
		t.Fatal(err)
	}
	p, err := NewPool(path)
	if err != nil {
		t.Fatal(err)
	}
	p.MarkKeyHeaderTimeout("k1")
	p.MarkKeyHeaderTimeout("k1")

	reloaded, err := NewPool(path)
	if err != nil {
		t.Fatal(err)
	}
	if got := reloaded.keys["k1"].ConsecutiveHeaderTimeouts; got != 2 {
		t.Fatalf("restored consecutive header timeouts = %d, want 2", got)
	}
	remaining := time.Until(reloaded.keys["k1"].CooldownUntil)
	if remaining < 14*time.Minute {
		t.Fatalf("restored bench remaining = %s, want the escalated ~15m", remaining)
	}
}

func TestPoolAbortKeyDoesNotCountClientCancellationAsError(t *testing.T) {
	config := `{"port":8080,"upstream_url":"http://127.0.0.1:1","max_queue_seconds":1,"max_concurrent_requests":2,"keys":[{"id":"k1","name":"K1","key":"test-key","rpm_limit":10,"enabled":true}]}`
	path := filepath.Join(t.TempDir(), "config.json")
	if err := os.WriteFile(path, []byte(config), 0600); err != nil {
		t.Fatal(err)
	}
	p, err := NewPool(path)
	if err != nil {
		t.Fatal(err)
	}
	key, _, err := p.AcquireKey(context.Background(), nil)
	if err != nil {
		t.Fatal(err)
	}
	p.AbortKey(key.Config.ID)
	if got := p.keys["k1"].ErrorCount; got != 0 {
		t.Fatalf("error count after abort = %d, want 0", got)
	}
	if got := p.keys["k1"].InFlightRequests; got != 0 {
		t.Fatalf("in-flight count after abort = %d, want 0", got)
	}
	if got := p.GetStats().ActiveRequests; got != 0 {
		t.Fatalf("active requests after abort = %d, want 0", got)
	}
}
