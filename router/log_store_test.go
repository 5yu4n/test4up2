package router

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestRequestLogStoreLoadsNewestCompleteRecords(t *testing.T) {
	path := filepath.Join(t.TempDir(), "requests.jsonl")
	for i := 1; i <= 5; i++ {
		if err := appendRequestLog(path, RequestLogItem{ID: fmt.Sprintf("req-%d", i), StatusCode: 200}); err != nil {
			t.Fatal(err)
		}
	}
	f, err := os.OpenFile(path, os.O_APPEND|os.O_WRONLY, requestLogFileMode)
	if err != nil {
		t.Fatal(err)
	}
	_, _ = f.WriteString("{interrupted\n")
	_ = f.Close()

	logs, err := loadRecentRequestLogs(path, 3)
	if err != nil {
		t.Fatal(err)
	}
	if len(logs) != 3 || logs[0].ID != "req-3" || logs[2].ID != "req-5" {
		t.Fatalf("loaded logs = %#v, want req-3..req-5", logs)
	}
}

func TestRequestLogStoreCompactsToRetentionLimit(t *testing.T) {
	path := filepath.Join(t.TempDir(), "requests.jsonl")
	for i := 1; i <= 6; i++ {
		if err := appendRequestLog(path, RequestLogItem{ID: fmt.Sprintf("req-%d", i), Timestamp: time.Now()}); err != nil {
			t.Fatal(err)
		}
	}
	if err := compactRequestLogs(path, 2); err != nil {
		t.Fatal(err)
	}
	logs, err := loadRecentRequestLogs(path, 10)
	if err != nil {
		t.Fatal(err)
	}
	if len(logs) != 2 || logs[0].ID != "req-5" || logs[1].ID != "req-6" {
		t.Fatalf("compacted logs = %#v, want req-5 and req-6", logs)
	}
}
