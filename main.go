package main

import (
	"embed"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"strings"

	"tokenrouter/router"
)

//go:embed web/index.html
var webFS embed.FS

func main() {
	configPath := "config.json"
	if len(os.Args) > 1 {
		configPath = os.Args[1]
	}

	pool, err := router.NewPool(configPath)
	if err != nil {
		log.Fatalf("Failed to initialize TokenRouter pool: %v", err)
	}

	proxyHandler := router.NewProxyHandler(pool)

	mux := http.NewServeMux()

	// Web Dashboard Static UI
	mux.HandleFunc("/dashboard", func(w http.ResponseWriter, r *http.Request) {
		data, err := webFS.ReadFile("web/index.html")
		if err != nil {
			http.Error(w, "Dashboard UI not found", http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		w.Header().Set("Cache-Control", "no-store")
		_, _ = w.Write(data)
	})

	mux.HandleFunc("/favicon.ico", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNoContent)
	})

	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/" || r.URL.Path == "/index.html" {
			http.Redirect(w, r, "/dashboard", http.StatusFound)
			return
		}

		p := r.URL.Path
		// Only forward genuine AI API requests to the proxy and key rotation pool
		if strings.HasPrefix(p, "/v1/") || strings.HasPrefix(p, "/messages") || strings.HasPrefix(p, "/chat/") || strings.HasPrefix(p, "/models") || strings.HasPrefix(p, "/embeddings") {
			proxyHandler.ServeHTTP(w, r)
			return
		}

		// Discard non-API scans / crawler requests cleanly without touching keys or logging fake requests
		http.NotFound(w, r)
	})

	// Dashboard Management APIs
	mux.HandleFunc("/api/stats", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		stats := pool.GetStats()
		_ = json.NewEncoder(w).Encode(stats)
	})

	mux.HandleFunc("/api/logs", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		logs := proxyHandler.GetLogs()
		_ = json.NewEncoder(w).Encode(logs)
	})

	mux.HandleFunc("/api/keys/toggle", func(w http.ResponseWriter, r *http.Request) {
		keyID := r.URL.Query().Get("id")
		enabled := r.URL.Query().Get("enabled") == "true"
		if keyID == "" {
			http.Error(w, `{"error":"missing key id"}`, http.StatusBadRequest)
			return
		}
		_ = pool.ToggleKey(keyID, enabled)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"status":"ok"}`))
	})

	mux.HandleFunc("/api/config/upstream", func(w http.ResponseWriter, r *http.Request) {
		urlStr := r.URL.Query().Get("url")
		if urlStr == "" {
			http.Error(w, `{"error":"missing url"}`, http.StatusBadRequest)
			return
		}
		_ = pool.SetUpstreamURL(urlStr)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"status":"ok"}`))
	})

	mux.HandleFunc("/api/config/freebuff", func(w http.ResponseWriter, r *http.Request) {
		enabled := r.URL.Query().Get("enabled") == "true"
		if err := pool.SetFreebuffEnabled(enabled); err != nil {
			http.Error(w, fmt.Sprintf(`{"error":%q}`, err.Error()), http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"status":"ok","freebuff_enabled":` + fmt.Sprintf("%t", enabled) + `}`))
	})

	mux.HandleFunc("/api/config/freebuff-url", func(w http.ResponseWriter, r *http.Request) {
		urlStr := r.URL.Query().Get("url")
		if urlStr == "" {
			http.Error(w, `{"error":"missing url"}`, http.StatusBadRequest)
			return
		}
		if err := pool.SetFreebuffBaseURL(urlStr); err != nil {
			http.Error(w, fmt.Sprintf(`{"error":%q}`, err.Error()), http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"status":"ok"}`))
	})

	mux.HandleFunc("/api/config/strip-thinking", func(w http.ResponseWriter, r *http.Request) {
		enabled := r.URL.Query().Get("enabled") == "true"
		_ = pool.SetStripThinking(enabled)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"status":"ok"}`))
	})

	mux.HandleFunc("/api/config/fast-thinking", func(w http.ResponseWriter, r *http.Request) {
		enabled := r.URL.Query().Get("enabled") == "true"
		_ = pool.SetFastStreamThinkingAsText(enabled)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"status":"ok"}`))
	})

	mux.HandleFunc("/api/config/effort", func(w http.ResponseWriter, r *http.Request) {
		effort := r.URL.Query().Get("effort")
		// "caller" disables the router-side default so every request keeps
		// exactly the effort behavior its own client asked for.
		if effort != "low" && effort != "high" && effort != "max" && effort != "caller" {
			effort = "high"
		}
		_ = pool.SetDefaultEffort(effort)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(fmt.Sprintf(`{"status":"ok","default_effort":%q}`, effort)))
	})


	mux.HandleFunc("/api/config/model-mapping", func(w http.ResponseWriter, r *http.Request) {
		alias := r.URL.Query().Get("alias")
		target := r.URL.Query().Get("target")
		if alias == "" {
			http.Error(w, `{"error":"missing alias"}`, http.StatusBadRequest)
			return
		}
		_ = pool.SetModelMapping(alias, target)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"status":"ok"}`))
	})

	port := pool.GetPort()
	addr := fmt.Sprintf(":%d", port)

	fmt.Println("==================================================================")
	fmt.Println("  ⚡ TokenRouter (Golang 本地配平限流器) 已經成功啟動！")
	fmt.Println("==================================================================")
	fmt.Printf("  • 服務代理 Port : http://localhost:%d/v1\n", port)
	fmt.Printf("  • Web 看板 UI   : http://localhost:%d/dashboard\n", port)
	stats := pool.GetStats()
	fmt.Printf("  • 當前金鑰總數  : %d 組 (總容量 %d RPM)\n", stats.TotalKeys, stats.AggregateMaxRPM)
	fmt.Printf("  • 上游 API 網址 : %s\n", pool.GetUpstreamURL())
	fmt.Printf("  • Freebuff Flash: %s (%s)\n", pool.GetFreebuffBaseURL(), map[bool]string{true: "enabled", false: "disabled"}[pool.IsFreebuffEnabled()])
	fmt.Println("==================================================================")

	if err := http.ListenAndServe(addr, mux); err != nil {
		log.Fatalf("Server exited with error: %v", err)
	}
}
