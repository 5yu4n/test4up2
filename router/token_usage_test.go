package router

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestParseTokenUsageSupportsProviderFieldNames(t *testing.T) {
	body := []byte(`{"usage":{"prompt_tokens":123,"completion_tokens":45,"total_tokens":168}}`)
	got := parseTokenUsage(body)
	if !got.InputKnown || got.InputTokens != 123 {
		t.Fatalf("input usage = %#v, want 123 exact", got)
	}
	if !got.OutputKnown || got.OutputTokens != 45 {
		t.Fatalf("output usage = %#v, want 45 exact", got)
	}

	camel := parseTokenUsage([]byte(`{"usage":{"inputTokens":7,"outputTokens":9}}`))
	if !camel.InputKnown || camel.InputTokens != 7 || !camel.OutputKnown || camel.OutputTokens != 9 {
		t.Fatalf("camel-case usage = %#v, want 7/9 exact", camel)
	}
}

func TestGenericSSEUsageSurvivesNetworkChunkBoundaries(t *testing.T) {
	tracker := &genericSSETerminalTracker{}
	tracker.Feed([]byte("data: {\"choices\":[],\"usage\":{\"prompt_tokens\":12,\"completion_tokens\":"))
	tracker.Feed([]byte("8}}\n\ndata: {\"choices\":[{\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n"))
	if err := tracker.Finish(); err != nil {
		t.Fatalf("Finish: %v", err)
	}
	got := tracker.Usage()
	if !got.InputKnown || got.InputTokens != 12 || !got.OutputKnown || got.OutputTokens != 8 {
		t.Fatalf("SSE usage = %#v, want 12/8 exact", got)
	}
}

func TestTokenUsageStoreMigratesHistoryAndPersistsNewMeasurements(t *testing.T) {
	path := filepath.Join(t.TempDir(), "usage.json")
	historical := []RequestLogItem{
		{StatusCode: 200, Path: "/v1/messages", KeyID: "key-1", RequestBytes: 400, BytesOut: 80, Timestamp: time.Now().Add(-time.Hour)},
		{StatusCode: 429, Path: "/v1/messages", KeyID: "key-2", RequestBytes: 1000, BytesOut: 1000},
		{StatusCode: 200, Path: "/v1/models", KeyID: "key-2", RequestBytes: 1000, BytesOut: 1000},
	}
	store := newTokenUsageStore(path, historical)
	got := store.snapshot()
	if got.Requests != 1 || got.ExactRequests != 0 || got.EstimatedRequests != 1 {
		t.Fatalf("migrated counters = %#v, want one estimated request", got)
	}
	if got.InputTokens != 100 || got.OutputTokens != 20 || got.TotalTokens != 120 {
		t.Fatalf("migrated tokens = %#v, want 100/20/120", got)
	}
	if got.ByKey["key-1"].Requests != 1 {
		t.Fatalf("per-key migration = %#v, want key-1 request", got.ByKey)
	}

	store.add("key-1", tokenUsageMeasurement{InputTokens: 30, OutputTokens: 10})
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var persisted tokenUsageState
	if err := json.Unmarshal(data, &persisted); err != nil {
		t.Fatalf("persisted usage JSON: %v", err)
	}
	if persisted.Stats.Requests != 2 || persisted.Stats.ExactRequests != 1 || persisted.Stats.EstimatedRequests != 1 {
		t.Fatalf("persisted counters = %#v, want exact+estimated", persisted.Stats)
	}
	if persisted.Stats.TotalTokens != 160 {
		t.Fatalf("persisted total = %d, want 160", persisted.Stats.TotalTokens)
	}
}
