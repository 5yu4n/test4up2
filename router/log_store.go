package router

import (
	"bufio"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"sync"
)

const (
	defaultRequestLogPath = "token-router-requests.jsonl"
	requestLogFileMode    = 0o600
	requestLogRotateBytes = 8 * 1024 * 1024
	requestLogKeepRecords = 2000
)

// requestLogSink serializes diagnostic writes. Request logs are written after
// the response body has been handed to net/http, and keeping the write here
// makes the latest record durable before a restart or a short-lived test
// process exits; no background writer can recreate a deleted deployment
// directory after shutdown.
type requestLogSink struct {
	mu               sync.Mutex
	path             string
	writesSinceCheck int
}

func newRequestLogSink(path string) *requestLogSink {
	if path == "" {
		path = defaultRequestLogPath
	}
	s := &requestLogSink{
		path: path,
	}
	return s
}

func initializeRequestLogs(pool *Pool, memoryLimit int) ([]RequestLogItem, *requestLogSink) {
	path := defaultRequestLogPath
	if pool != nil && pool.configPath != "" {
		// Keeping diagnostics beside the selected config isolates test instances
		// and makes alternate deployments retain their own history.
		path = pool.configPath + ".requests.jsonl"
	}
	logs, err := loadRecentRequestLogs(path, memoryLimit)
	if err != nil {
		logs = nil
	}
	return logs, newRequestLogSink(path)
}

func (s *requestLogSink) enqueue(item RequestLogItem) {
	if s == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	_ = appendRequestLog(s.path, item)
	s.writesSinceCheck++
	if s.writesSinceCheck >= 100 {
		s.writesSinceCheck = 0
		if info, err := os.Stat(s.path); err == nil && info.Size() > requestLogRotateBytes {
			_ = compactRequestLogs(s.path, requestLogKeepRecords)
		}
	}
}

func appendRequestLog(path string, item RequestLogItem) error {
	data, err := json.Marshal(item)
	if err != nil {
		return err
	}
	if dir := filepath.Dir(path); dir != "." && dir != "" {
		if err := os.MkdirAll(dir, 0o700); err != nil {
			return err
		}
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, requestLogFileMode)
	if err != nil {
		return err
	}
	defer f.Close()
	if _, err := f.Write(append(data, '\n')); err != nil {
		return err
	}
	return nil
}

func loadRecentRequestLogs(path string, limit int) ([]RequestLogItem, error) {
	if limit <= 0 {
		return nil, nil
	}
	f, err := os.Open(path)
	if errors.Is(err, os.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	defer f.Close()

	logs := make([]RequestLogItem, 0, limit)
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 64*1024), 2*1024*1024)
	for scanner.Scan() {
		var item RequestLogItem
		if json.Unmarshal(scanner.Bytes(), &item) != nil {
			// A process can be interrupted during its final append. Preserve every
			// complete record around that partial line.
			continue
		}
		if len(logs) == limit {
			copy(logs, logs[1:])
			logs[len(logs)-1] = item
		} else {
			logs = append(logs, item)
		}
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	return logs, nil
}

func compactRequestLogs(path string, keep int) error {
	logs, err := loadRecentRequestLogs(path, keep)
	if err != nil {
		return err
	}
	tmp := path + ".tmp"
	f, err := os.OpenFile(tmp, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, requestLogFileMode)
	if err != nil {
		return err
	}
	encoder := json.NewEncoder(f)
	for _, item := range logs {
		if err := encoder.Encode(item); err != nil {
			_ = f.Close()
			_ = os.Remove(tmp)
			return err
		}
	}
	if err := f.Close(); err != nil {
		_ = os.Remove(tmp)
		return err
	}
	if err := os.Rename(tmp, path); err != nil {
		_ = os.Remove(path)
		if retryErr := os.Rename(tmp, path); retryErr != nil {
			_ = os.Remove(tmp)
			return retryErr
		}
	}
	return nil
}
