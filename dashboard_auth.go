package main

import (
	"crypto/rand"
	"encoding/base64"
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"
)

const dashboardSessionCookie = "tokenrouter_dashboard_session"

type dashboardAuth struct {
	mu       sync.Mutex
	sessions map[string]time.Time
	ttl      time.Duration
}

func newDashboardAuth() *dashboardAuth {
	return &dashboardAuth{
		sessions: make(map[string]time.Time),
		ttl:      12 * time.Hour,
	}
}

func (a *dashboardAuth) cleanupLocked(now time.Time) {
	for token, expiresAt := range a.sessions {
		if !expiresAt.After(now) {
			delete(a.sessions, token)
		}
	}
}

func (a *dashboardAuth) authenticated(r *http.Request) bool {
	cookie, err := r.Cookie(dashboardSessionCookie)
	if err != nil || cookie.Value == "" {
		return false
	}

	now := time.Now()
	a.mu.Lock()
	defer a.mu.Unlock()
	a.cleanupLocked(now)
	expiresAt, ok := a.sessions[cookie.Value]
	if !ok || !expiresAt.After(now) {
		return false
	}
	// Keep an active administrator session alive without making it permanent.
	a.sessions[cookie.Value] = now.Add(a.ttl)
	return true
}

func (a *dashboardAuth) require(w http.ResponseWriter, r *http.Request) bool {
	if a.authenticated(r) {
		return true
	}
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Cache-Control", "no-store")
	w.WriteHeader(http.StatusUnauthorized)
	_, _ = w.Write([]byte(`{"error":"dashboard authentication required"}`))
	return false
}

func (a *dashboardAuth) issueSession(w http.ResponseWriter, r *http.Request) error {
	raw := make([]byte, 32)
	if _, err := rand.Read(raw); err != nil {
		return err
	}
	token := base64.RawURLEncoding.EncodeToString(raw)
	now := time.Now()
	a.mu.Lock()
	a.cleanupLocked(now)
	a.sessions[token] = now.Add(a.ttl)
	a.mu.Unlock()

	secure := r.TLS != nil || strings.EqualFold(r.Header.Get("X-Forwarded-Proto"), "https")
	http.SetCookie(w, &http.Cookie{
		Name:     dashboardSessionCookie,
		Value:    token,
		Path:     "/",
		MaxAge:   int(a.ttl.Seconds()),
		HttpOnly: true,
		Secure:   secure,
		SameSite: http.SameSiteStrictMode,
	})
	return nil
}

func (a *dashboardAuth) clearSession(w http.ResponseWriter, r *http.Request) {
	if cookie, err := r.Cookie(dashboardSessionCookie); err == nil {
		a.mu.Lock()
		delete(a.sessions, cookie.Value)
		a.mu.Unlock()
	}
	secure := r.TLS != nil || strings.EqualFold(r.Header.Get("X-Forwarded-Proto"), "https")
	http.SetCookie(w, &http.Cookie{
		Name:     dashboardSessionCookie,
		Value:    "",
		Path:     "/",
		MaxAge:   -1,
		HttpOnly: true,
		Secure:   secure,
		SameSite: http.SameSiteStrictMode,
	})
}

func decodeDashboardPassword(r *http.Request) (string, error) {
	var payload struct {
		Password string `json:"password"`
	}
	decoder := json.NewDecoder(io.LimitReader(r.Body, 4096))
	if err := decoder.Decode(&payload); err != nil {
		return "", err
	}
	return payload.Password, nil
}
