package router

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

func TestAnthropicRequestToOpenAI(t *testing.T) {
	body, err := anthropicRequestToOpenAI([]byte(`{"model":"z-ai/glm-5.3-free","system":"Be concise.","max_tokens":32,"stop_sequences":["END"],"thinking":{"type":"enabled"},"messages":[{"role":"user","content":[{"type":"text","text":"hello"}]}]}`))
	if err != nil {
		t.Fatal(err)
	}
	var got map[string]interface{}
	if err := json.Unmarshal(body, &got); err != nil {
		t.Fatal(err)
	}
	if _, ok := got["reasoning_effort"]; ok {
		t.Fatalf("reasoning_effort was injected into an unspecified request")
	}
	if got["max_tokens"] != float64(32) {
		t.Fatalf("max_tokens = %v, want 32", got["max_tokens"])
	}
	if _, ok := got["thinking"]; ok {
		t.Fatal("Anthropic thinking envelope leaked to OpenAI request")
	}
	messages, ok := got["messages"].([]interface{})
	if !ok || len(messages) != 2 {
		t.Fatalf("messages = %#v, want system + user", got["messages"])
	}
	if messages[0].(map[string]interface{})["role"] != "system" {
		t.Fatalf("first message role = %#v", messages[0])
	}
}

func TestAnthropicToolNamesUseStableOpenAIAliases(t *testing.T) {
	longName := "mcp__plugin_android-emulator_android-emulator__android_discover_project"
	body := `{"model":"z-ai/glm-5.3-free","max_tokens":4,"stream":true,"tools":[{"name":"` + longName + `","description":"discover","input_schema":{"type":"object"}}],"tool_choice":{"type":"tool","name":"` + longName + `"},"messages":[{"role":"assistant","content":[{"type":"tool_use","id":"call_1","name":"` + longName + `","input":{"path":"."}}]},{"role":"user","content":[{"type":"tool_result","tool_use_id":"call_1","content":"ok"}]}]}`

	converted, aliases, err := anthropicRequestToOpenAIWithAliases([]byte(body))
	if err != nil {
		t.Fatal(err)
	}
	alias := aliases.forward[longName]
	if alias == "" || alias == longName || !isOpenAIToolName(alias) {
		t.Fatalf("alias = %q, want a distinct valid OpenAI name", alias)
	}

	var got map[string]interface{}
	if err := json.Unmarshal(converted, &got); err != nil {
		t.Fatal(err)
	}
	tools := got["tools"].([]interface{})
	toolFn := tools[0].(map[string]interface{})["function"].(map[string]interface{})
	if toolFn["name"] != alias {
		t.Fatalf("tool definition name = %v, want %q", toolFn["name"], alias)
	}
	choice := got["tool_choice"].(map[string]interface{})["function"].(map[string]interface{})
	if choice["name"] != alias {
		t.Fatalf("tool choice name = %v, want %q", choice["name"], alias)
	}
	messages := got["messages"].([]interface{})
	call := messages[0].(map[string]interface{})["tool_calls"].([]interface{})[0].(map[string]interface{})
	callFn := call["function"].(map[string]interface{})
	if callFn["name"] != alias {
		t.Fatalf("historical tool call name = %v, want %q", callFn["name"], alias)
	}

	response := `{"id":"chat_1","model":"z-ai/glm-5.3-free","choices":[{"message":{"tool_calls":[{"id":"call_1","function":{"name":"` + alias + `","arguments":"{\"path\":\".\"}"}}]},"finish_reason":"tool_calls"}]}`
	restored, err := openAIJSONToAnthropicWithAliases([]byte(response), "fallback", aliases)
	if err != nil {
		t.Fatal(err)
	}
	var restoredEnvelope map[string]interface{}
	if err := json.Unmarshal(restored, &restoredEnvelope); err != nil {
		t.Fatal(err)
	}
	restoredContent := restoredEnvelope["content"].([]interface{})
	restoredTool := restoredContent[0].(map[string]interface{})
	if restoredTool["name"] != longName {
		t.Fatalf("restored tool name = %v, want original name", restoredTool["name"])
	}
}

func TestAnthropicStreamConverterEmitsParallelToolCalls(t *testing.T) {
	var out bytes.Buffer
	converter := newAnthropicStreamConverter(func(chunk []byte) error {
		_, err := out.Write(chunk)
		return err
	}, "glm", "msg_1")

	feed := func(payload string) {
		if err := converter.consumeData([]byte(payload)); err != nil {
			t.Fatalf("consumeData(%s): %v", payload, err)
		}
	}
	// Two parallel calls; each call's id/name arrive only in its first delta,
	// and the gateway interleaves the argument fragments.
	feed(`{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_a","function":{"name":"read_file","arguments":""}}]}}]}`)
	feed(`{"choices":[{"delta":{"tool_calls":[{"index":1,"id":"call_b","function":{"name":"write_file","arguments":""}}]}}]}`)
	feed(`{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"path\":"}}]}}]}`)
	feed(`{"choices":[{"delta":{"tool_calls":[{"index":1,"function":{"arguments":"\"b.txt\""}}]}}]}`)
	feed(`{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"a.txt\"}"}}]}}]}`)
	feed(`{"choices":[{"delta":{"tool_calls":[{"index":1,"function":{"arguments":"{}"}}]}}]}`)
	feed(`{"choices":[{"delta":{},"finish_reason":"tool_calls"}]}`)
	feed(`[DONE]`)

	body := out.String()
	for _, want := range []string{
		`"name":"read_file"`,
		`"name":"write_file"`,
		`"id":"call_a"`,
		`"id":"call_b"`,
	} {
		if !strings.Contains(body, want) {
			t.Fatalf("stream missing %s:\n%s", want, body)
		}
	}
	// The first call streams live (its argument fragments arrive as separate
	// deltas on block 0, exactly as upstream sent them); the parallel call is
	// buffered and emitted complete on block 1.
	if !strings.Contains(body, `{"delta":{"partial_json":"{\"path\":","type":"input_json_delta"},"index":0`) {
		t.Fatalf("call_a first fragment missing on block 0:\n%s", body)
	}
	if !strings.Contains(body, `{"delta":{"partial_json":"\"a.txt\"}","type":"input_json_delta"},"index":0`) {
		t.Fatalf("call_a second fragment missing on block 0:\n%s", body)
	}
	if !strings.Contains(body, `{"delta":{"partial_json":"\"b.txt\"{}","type":"input_json_delta"},"index":1`) {
		t.Fatalf("call_b arguments not complete in block 1:\n%s", body)
	}
	if got := strings.Count(body, `"type":"content_block_start"`); got != 2 {
		t.Fatalf("content_block_start count = %d, want 2:\n%s", got, body)
	}
	if got := strings.Count(body, `"type":"content_block_stop"`); got != 2 {
		t.Fatalf("content_block_stop count = %d, want 2:\n%s", got, body)
	}
	if !strings.Contains(body, `"stop_reason":"tool_use"`) {
		t.Fatalf("stop_reason missing:\n%s", body)
	}
}

func TestAnthropicStreamConverterPreservesThinkingAndText(t *testing.T) {
	var output strings.Builder
	converter := newAnthropicStreamConverter(func(chunk []byte) error {
		output.Write(chunk)
		return nil
	}, "z-ai/glm-5.3-free", "msg_test")
	if err := converter.start(); err != nil {
		t.Fatal(err)
	}
	if err := converter.consumeData([]byte(`{"model":"z-ai/glm-5.3-free","choices":[{"delta":{"reasoning_content":"think"}}]}`)); err != nil {
		t.Fatal(err)
	}
	if err := converter.consumeData([]byte(`{"choices":[{"delta":{"content":"answer"},"finish_reason":"stop"}]}`)); err != nil {
		t.Fatal(err)
	}
	if err := converter.consumeData([]byte(`[DONE]`)); err != nil {
		t.Fatal(err)
	}
	result := output.String()
	for _, want := range []string{"event: message_start", "thinking_delta", "think", "text_delta", "answer", "event: message_stop"} {
		if !strings.Contains(result, want) {
			t.Fatalf("converted stream missing %q: %s", want, result)
		}
	}
}

func TestAnthropicStreamConverterStartsToolBlockAtZero(t *testing.T) {
	var output strings.Builder
	converter := newAnthropicStreamConverter(func(chunk []byte) error {
		output.Write(chunk)
		return nil
	}, "z-ai/glm-5.3-free", "msg_tool")
	if err := converter.start(); err != nil {
		t.Fatal(err)
	}
	if err := converter.consumeData([]byte(`{"choices":[{"delta":{"tool_calls":[{"id":"call_1","function":{"name":"noop","arguments":"{}"}}]}}]}`)); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(output.String(), `"index":0`) {
		t.Fatalf("first tool block did not use index 0: %s", output.String())
	}
}

func TestOpenAIJSONToAnthropic(t *testing.T) {
	converted, err := openAIJSONToAnthropic([]byte(`{"id":"chat_1","model":"z-ai/glm-5.3-free","choices":[{"message":{"reasoning_content":"think","content":"answer"},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":2}}`), "fallback")
	if err != nil {
		t.Fatal(err)
	}
	var got map[string]interface{}
	if err := json.Unmarshal(converted, &got); err != nil {
		t.Fatal(err)
	}
	if got["type"] != "message" || got["stop_reason"] != "end_turn" {
		t.Fatalf("unexpected response envelope: %#v", got)
	}
	content := got["content"].([]interface{})
	if len(content) != 2 || content[0].(map[string]interface{})["type"] != "thinking" || content[1].(map[string]interface{})["type"] != "text" {
		t.Fatalf("content = %#v", content)
	}
}
