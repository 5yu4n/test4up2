package router

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
)

// tokenRouterAnthropicBridge is enabled only for TokenRouter's /messages
// compatibility path. TokenRouter documents the GLM free endpoint as
// OpenAI-compatible, so translating at the edge avoids the slower/less stable
// Anthropic compatibility route while keeping Claude clients unchanged.
func tokenRouterAnthropicBridge(host, path string) bool {
	host = strings.ToLower(strings.TrimSpace(host))
	return strings.Contains(host, "api.tokenrouter.com") && strings.Contains(path, "/messages")
}

func isTokenRouterHost(host string) bool {
	return strings.Contains(strings.ToLower(strings.TrimSpace(host)), "api.tokenrouter.com")
}

const maxOpenAIToolNameLength = 64

// toolNameAliases keeps Claude/Anthropic tool names stable at the edge while
// satisfying OpenAI-compatible gateways, which cap function names at 64
// ASCII characters. The mapping is per request so historical tool calls and
// the current tool definitions always use the same alias.
type toolNameAliases struct {
	forward map[string]string
	reverse map[string]string
}

func isOpenAIToolName(name string) bool {
	if name == "" || len(name) > maxOpenAIToolNameLength {
		return false
	}
	for i := 0; i < len(name); i++ {
		c := name[i]
		if (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
			(c >= '0' && c <= '9') || c == '_' || c == '-' {
			continue
		}
		return false
	}
	return true
}

func newToolNameAliases(names []string) *toolNameAliases {
	aliases := &toolNameAliases{
		forward: make(map[string]string),
		reverse: make(map[string]string),
	}
	seen := make(map[string]struct{}, len(names))
	for _, name := range names {
		if name == "" {
			continue
		}
		if _, ok := seen[name]; ok {
			continue
		}
		seen[name] = struct{}{}
		// Reserve every already-valid name first. This prevents a generated
		// alias from colliding with a real short tool name.
		if isOpenAIToolName(name) {
			aliases.forward[name] = name
			aliases.reverse[name] = name
		}
	}
	for name := range seen {
		if !isOpenAIToolName(name) {
			aliases.mapName(name)
		}
	}
	return aliases
}

func (a *toolNameAliases) mapName(name string) string {
	if a == nil || name == "" {
		return name
	}
	if alias, ok := a.forward[name]; ok {
		return alias
	}
	if isOpenAIToolName(name) {
		a.forward[name] = name
		a.reverse[name] = name
		return name
	}

	// A short hash is deterministic across retries and preserves no user data
	// in the wire name. The collision loop is defensive for the unlikely case
	// that a real tool already occupies the generated name.
	sum := sha256.Sum256([]byte(name))
	base := "tr_" + hex.EncodeToString(sum[:])[:32]
	alias := base
	for suffix := 1; ; suffix++ {
		if owner, exists := a.reverse[alias]; !exists || owner == name {
			a.forward[name] = alias
			a.reverse[alias] = name
			return alias
		}
		ending := fmt.Sprintf("_%d", suffix)
		prefixLen := maxOpenAIToolNameLength - len(ending)
		if prefixLen < 1 {
			prefixLen = 1
		}
		alias = base[:minInt(len(base), prefixLen)] + ending
	}
}

func (a *toolNameAliases) restoreName(name string) string {
	if a == nil {
		return name
	}
	if original, ok := a.reverse[name]; ok {
		return original
	}
	return name
}

func minInt(left, right int) int {
	if left < right {
		return left
	}
	return right
}

func collectAnthropicToolNames(in map[string]interface{}) []string {
	names := make([]string, 0)
	if rawTools, ok := in["tools"].([]interface{}); ok {
		for _, raw := range rawTools {
			if tool, ok := raw.(map[string]interface{}); ok {
				if name, ok := tool["name"].(string); ok {
					names = append(names, name)
				}
			}
		}
	}
	if choice, ok := in["tool_choice"].(map[string]interface{}); ok {
		if name, ok := choice["name"].(string); ok {
			names = append(names, name)
		}
	}
	if rawMessages, ok := in["messages"].([]interface{}); ok {
		for _, raw := range rawMessages {
			message, ok := raw.(map[string]interface{})
			if !ok {
				continue
			}
			blocks, ok := message["content"].([]interface{})
			if !ok {
				continue
			}
			for _, rawBlock := range blocks {
				block, ok := rawBlock.(map[string]interface{})
				if !ok || block["type"] != "tool_use" {
					continue
				}
				if name, ok := block["name"].(string); ok {
					names = append(names, name)
				}
			}
		}
	}
	return names
}

func anthropicTextContent(value interface{}) string {
	if text, ok := value.(string); ok {
		return text
	}
	blocks, ok := value.([]interface{})
	if !ok {
		return ""
	}
	var b strings.Builder
	for _, raw := range blocks {
		block, ok := raw.(map[string]interface{})
		if !ok {
			continue
		}
		typ, _ := block["type"].(string)
		switch typ {
		case "text":
			if text, ok := block["text"].(string); ok {
				b.WriteString(text)
			}
		case "thinking":
			if thinking, ok := block["thinking"].(string); ok {
				b.WriteString(thinking)
			}
		}
	}
	return b.String()
}

func anthropicContentToOpenAI(value interface{}) interface{} {
	if _, ok := value.(string); ok {
		return value
	}
	blocks, ok := value.([]interface{})
	if !ok {
		return value
	}
	allText := true
	textValue := anthropicTextContent(value)
	mapped := make([]interface{}, 0, len(blocks))
	for _, raw := range blocks {
		block, ok := raw.(map[string]interface{})
		if !ok {
			allText = false
			continue
		}
		typ, _ := block["type"].(string)
		switch typ {
		case "text":
			mapped = append(mapped, map[string]interface{}{"type": "text", "text": block["text"]})
		case "image":
			// Anthropic image sources map cleanly to OpenAI image_url blocks.
			if source, ok := block["source"].(map[string]interface{}); ok {
				if data, ok := source["data"].(string); ok {
					mediaType, _ := source["media_type"].(string)
					if mediaType == "" {
						mediaType = "application/octet-stream"
					}
					mapped = append(mapped, map[string]interface{}{
						"type":      "image_url",
						"image_url": map[string]interface{}{"url": "data:" + mediaType + ";base64," + data},
					})
					allText = false
					continue
				}
			}
			allText = false
		case "thinking":
			// Thinking supplied in prior assistant turns is retained as text
			// context; the current request's reasoning_effort controls new work.
			mapped = append(mapped, map[string]interface{}{"type": "text", "text": block["thinking"]})
		default:
			allText = false
		}
	}
	if allText {
		return textValue
	}
	if len(mapped) > 0 {
		return mapped
	}
	return textValue
}

func anthropicToolsToOpenAI(value interface{}) interface{} {
	return anthropicToolsToOpenAIWithAliases(value, nil)
}

func anthropicToolsToOpenAIWithAliases(value interface{}, aliases *toolNameAliases) interface{} {
	rawTools, ok := value.([]interface{})
	if !ok {
		return value
	}
	tools := make([]interface{}, 0, len(rawTools))
	for _, raw := range rawTools {
		tool, ok := raw.(map[string]interface{})
		if !ok {
			continue
		}
		fn := map[string]interface{}{}
		for _, key := range []string{"name", "description"} {
			if value, exists := tool[key]; exists {
				if key == "name" {
					if name, ok := value.(string); ok {
						value = mapToolName(aliases, name)
					}
				}
				fn[key] = value
			}
		}
		if schema, exists := tool["input_schema"]; exists {
			fn["parameters"] = schema
		} else if schema, exists := tool["parameters"]; exists {
			fn["parameters"] = schema
		}
		tools = append(tools, map[string]interface{}{"type": "function", "function": fn})
	}
	return tools
}

func anthropicToolChoiceToOpenAI(value interface{}) interface{} {
	return anthropicToolChoiceToOpenAIWithAliases(value, nil)
}

func anthropicToolChoiceToOpenAIWithAliases(value interface{}, aliases *toolNameAliases) interface{} {
	choice, ok := value.(map[string]interface{})
	if !ok {
		if text, ok := value.(string); ok && text == "any" {
			return "required"
		}
		return value
	}
	typ, _ := choice["type"].(string)
	switch typ {
	case "any":
		return "required"
	case "auto":
		return "auto"
	case "tool":
		if name, ok := choice["name"].(string); ok {
			return map[string]interface{}{"type": "function", "function": map[string]interface{}{"name": mapToolName(aliases, name)}}
		}
	}
	return value
}

func anthropicMessagesToOpenAI(value interface{}) []interface{} {
	return anthropicMessagesToOpenAIWithAliases(value, nil)
}

func anthropicMessagesToOpenAIWithAliases(value interface{}, aliases *toolNameAliases) []interface{} {
	rawMessages, ok := value.([]interface{})
	if !ok {
		return nil
	}
	messages := make([]interface{}, 0, len(rawMessages))
	for _, raw := range rawMessages {
		message, ok := raw.(map[string]interface{})
		if !ok {
			continue
		}
		role, _ := message["role"].(string)
		content := message["content"]
		blocks, isBlocks := content.([]interface{})
		if !isBlocks {
			copyMessage := make(map[string]interface{}, len(message))
			for key, item := range message {
				copyMessage[key] = item
			}
			messages = append(messages, copyMessage)
			continue
		}

		textParts := make([]string, 0)
		openAIContent := make([]interface{}, 0)
		toolCalls := make([]interface{}, 0)
		toolResults := make([]interface{}, 0)
		for _, blockRaw := range blocks {
			block, ok := blockRaw.(map[string]interface{})
			if !ok {
				continue
			}
			typ, _ := block["type"].(string)
			switch typ {
			case "text":
				if text, ok := block["text"].(string); ok {
					textParts = append(textParts, text)
				}
			case "image":
				if converted, ok := anthropicContentToOpenAI([]interface{}{block}).([]interface{}); ok {
					openAIContent = append(openAIContent, converted...)
				}
			case "tool_use":
				if role == "assistant" {
					arguments, _ := json.Marshal(block["input"])
					toolCalls = append(toolCalls, map[string]interface{}{
						"id": block["id"], "type": "function",
						"function": map[string]interface{}{"name": mapToolName(aliases, stringValue(block["name"])), "arguments": string(arguments)},
					})
				}
			case "tool_result":
				if role == "user" {
					toolResults = append(toolResults, map[string]interface{}{
						"role": "tool", "tool_call_id": block["tool_use_id"], "content": anthropicTextContent(block["content"]),
					})
				}
			case "thinking":
				// Historical reasoning is provider-internal state, not user context.
				// Replaying it as ordinary text inflated the failing project request
				// by hundreds of thousands of bytes and pushed the OpenAI-compatible
				// model past its practical context/cache admission path. The current
				// turn's reasoning is still streamed back by the response converter.
			}
		}
		copyMessage := map[string]interface{}{"role": role}
		if len(openAIContent) > 0 {
			copyMessage["content"] = append([]interface{}{}, openAIContent...)
			if len(textParts) > 0 {
				copyMessage["content"] = append([]interface{}{map[string]interface{}{"type": "text", "text": strings.Join(textParts, "")}}, openAIContent...)
			}
		} else if len(textParts) > 0 {
			copyMessage["content"] = strings.Join(textParts, "")
		} else if len(toolCalls) > 0 {
			copyMessage["content"] = nil
		}
		if len(toolCalls) > 0 {
			copyMessage["tool_calls"] = toolCalls
		}
		if len(toolResults) > 0 {
			if len(textParts) > 0 {
				copyMessage["content"] = strings.Join(textParts, "")
			}
			messages = append(messages, toolResults...)
			if len(textParts) == 0 {
				continue
			}
		}
		messages = append(messages, copyMessage)
	}
	return messages
}

func anthropicRequestToOpenAI(body []byte) ([]byte, error) {
	converted, _, err := anthropicRequestToOpenAIWithAliases(body)
	return converted, err
}

func anthropicRequestToOpenAIWithAliases(body []byte) ([]byte, *toolNameAliases, error) {
	var in map[string]interface{}
	if err := json.Unmarshal(body, &in); err != nil {
		return nil, nil, fmt.Errorf("invalid Anthropic request: %w", err)
	}
	aliases := newToolNameAliases(collectAnthropicToolNames(in))
	out := make(map[string]interface{}, len(in)+2)
	if model, ok := in["model"]; ok {
		out["model"] = model
	}
	if stream, ok := in["stream"]; ok {
		out["stream"] = stream
	}
	for _, key := range []string{"temperature", "top_p", "parallel_tool_calls", "reasoning_effort", "stream_options"} {
		if value, ok := in[key]; ok {
			out[key] = value
		}
	}
	if tools, ok := in["tools"]; ok {
		out["tools"] = anthropicToolsToOpenAIWithAliases(tools, aliases)
	}
	if choice, ok := in["tool_choice"]; ok {
		out["tool_choice"] = anthropicToolChoiceToOpenAIWithAliases(choice, aliases)
	}
	if maxTokens, ok := in["max_tokens"]; ok {
		out["max_tokens"] = maxTokens
	} else if maxTokens, ok := in["max_tokens_to_sample"]; ok {
		out["max_tokens"] = maxTokens
	}
	if stop, ok := in["stop_sequences"]; ok {
		out["stop"] = stop
	} else if stop, ok := in["stop"]; ok {
		out["stop"] = stop
	}
	messages := make([]interface{}, 0)
	if system, ok := in["system"]; ok {
		if text := anthropicTextContent(system); text != "" {
			messages = append(messages, map[string]interface{}{"role": "system", "content": text})
		}
	}
	if rawMessages, ok := in["messages"]; ok {
		messages = append(messages, anthropicMessagesToOpenAIWithAliases(rawMessages, aliases)...)
	}
	out["messages"] = messages
	converted, err := json.Marshal(out)
	if err != nil {
		return nil, nil, err
	}
	return converted, aliases, nil
}

func mapToolName(aliases *toolNameAliases, name string) string {
	if aliases == nil {
		return name
	}
	return aliases.mapName(name)
}

func stringValue(value interface{}) string {
	name, _ := value.(string)
	return name
}

type openAIChoice struct {
	Delta        map[string]interface{} `json:"delta"`
	FinishReason string                 `json:"finish_reason"`
	Message      map[string]interface{} `json:"message"`
}

type openAIChunk struct {
	ID      string                 `json:"id"`
	Model   string                 `json:"model"`
	Choices []openAIChoice         `json:"choices"`
	Usage   map[string]interface{} `json:"usage"`
}

type upstreamSSEError struct {
	Type    string
	Message string
}

func (e *upstreamSSEError) Error() string {
	if e.Type == "" {
		return "upstream stream error: " + e.Message
	}
	return fmt.Sprintf("upstream stream error (%s): %s", e.Type, e.Message)
}

// openAIStreamError recognizes application-level failures delivered inside an
// HTTP 200 SSE response. Treating these as an ordinary chunk followed by
// [DONE] would turn a provider error into a false empty success.
func openAIStreamError(data []byte) error {
	var envelope struct {
		Error json.RawMessage `json:"error"`
	}
	if json.Unmarshal(data, &envelope) != nil || len(envelope.Error) == 0 || string(envelope.Error) == "null" {
		return nil
	}
	var detail struct {
		Type    string `json:"type"`
		Code    string `json:"code"`
		Message string `json:"message"`
	}
	if json.Unmarshal(envelope.Error, &detail) == nil {
		if detail.Type == "" {
			detail.Type = detail.Code
		}
		if detail.Message != "" {
			return &upstreamSSEError{Type: detail.Type, Message: detail.Message}
		}
	}
	var message string
	if json.Unmarshal(envelope.Error, &message) == nil && message != "" {
		return &upstreamSSEError{Message: message}
	}
	return &upstreamSSEError{Message: strings.TrimSpace(string(envelope.Error))}
}

// toolCallBuffer accumulates one OpenAI tool_call across its argument
// deltas. Parallel calls are assembled per index and emitted as complete
// Anthropic blocks at finish: blocks are then strictly sequential, which the
// Anthropic streaming contract requires, and clients cannot execute tools
// before message_stop anyway, so nothing is lost by not interleaving.
type toolCallBuffer struct {
	id   string
	name string
	args strings.Builder
}

type anthropicStreamConverter struct {
	emit       func([]byte) error
	model      string
	messageID  string
	aliases    *toolNameAliases
	blockType  string
	blockIndex int
	toolName   string
	toolID     string
	stopReason string
	started    bool
	done       bool
	// liveToolIndex is the OpenAI tool_call index that streams live as an
	// Anthropic block (the first call seen). Parallel calls with different
	// indexes are buffered and emitted sequentially at finish.
	liveToolIndex int
	// toolCalls buffers each parallel tool call by its OpenAI index.
	toolCalls map[int]*toolCallBuffer
	// tokenUsage captures provider usage fields that may arrive in the final
	// OpenAI SSE chunk. TokenRouter's Anthropic response events historically
	// exposed zeroes, so keep the exact values for accounting even when the
	// client-facing compatibility event cannot be changed.
	tokenUsage parsedTokenUsage
}

func newAnthropicStreamConverter(emit func([]byte) error, model, messageID string) *anthropicStreamConverter {
	return newAnthropicStreamConverterWithAliases(emit, model, messageID, nil)
}

func newAnthropicStreamConverterWithAliases(emit func([]byte) error, model, messageID string, aliases *toolNameAliases) *anthropicStreamConverter {
	return &anthropicStreamConverter{emit: emit, model: model, messageID: messageID, aliases: aliases, liveToolIndex: -1, toolCalls: make(map[int]*toolCallBuffer)}
}

func (c *anthropicStreamConverter) event(name string, payload interface{}) error {
	data, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	chunk := append([]byte("event: "+name+"\ndata: "), data...)
	chunk = append(chunk, '\n', '\n')
	return c.emit(chunk)
}

func (c *anthropicStreamConverter) start() error {
	if c.started {
		return nil
	}
	c.started = true
	return c.event("message_start", map[string]interface{}{
		"type": "message_start",
		"message": map[string]interface{}{
			"id": c.messageID, "type": "message", "role": "assistant", "content": []interface{}{},
			"model": c.model, "stop_reason": nil, "stop_sequence": nil,
			"usage": map[string]interface{}{"input_tokens": 0, "output_tokens": 0},
		},
	})
}

func (c *anthropicStreamConverter) startBlock(blockType string, index int) error {
	if c.blockType != "" {
		if err := c.event("content_block_stop", map[string]interface{}{"type": "content_block_stop", "index": c.blockIndex}); err != nil {
			return err
		}
	}
	c.blockType = blockType
	c.blockIndex = index
	content := map[string]interface{}{"type": blockType}
	if blockType == "thinking" {
		content["thinking"] = ""
		content["signature"] = ""
	} else if blockType == "text" {
		content["text"] = ""
	} else if blockType == "tool_use" {
		content["id"] = c.toolID
		content["name"] = c.toolName
		content["input"] = map[string]interface{}{}
	}
	return c.event("content_block_start", map[string]interface{}{"type": "content_block_start", "index": index, "content_block": content})
}

func (c *anthropicStreamConverter) consumeData(data []byte) error {
	if c.done {
		return nil
	}
	if string(data) == "[DONE]" {
		if err := c.start(); err != nil {
			return err
		}
		return c.finish()
	}
	if err := openAIStreamError(data); err != nil {
		return err
	}
	var chunk openAIChunk
	if err := json.Unmarshal(data, &chunk); err != nil {
		return nil // Ignore provider keep-alives and non-JSON SSE lines.
	}
	if chunk.Model != "" {
		c.model = chunk.Model
	}
	mergeTokenUsage(&c.tokenUsage, parseTokenUsageMap(chunk.Usage))
	if len(chunk.Choices) == 0 {
		return nil
	}
	if err := c.start(); err != nil {
		return err
	}
	choice := chunk.Choices[0]
	if choice.FinishReason != "" {
		c.stopReason = choice.FinishReason
	}
	reasoning, _ := choice.Delta["reasoning_content"].(string)
	if reasoning == "" {
		reasoning, _ = choice.Delta["reasoning"].(string)
	}
	if reasoning != "" {
		if c.blockType != "thinking" {
			if err := c.startBlock("thinking", 0); err != nil {
				return err
			}
		}
		if err := c.event("content_block_delta", map[string]interface{}{"type": "content_block_delta", "index": c.blockIndex, "delta": map[string]interface{}{"type": "thinking_delta", "thinking": reasoning}}); err != nil {
			return err
		}
	}
	content, _ := choice.Delta["content"].(string)
	if content != "" {
		index := 0
		if c.blockType == "thinking" {
			index = 1
		}
		if c.blockType != "text" {
			if err := c.startBlock("text", index); err != nil {
				return err
			}
		}
		if err := c.event("content_block_delta", map[string]interface{}{"type": "content_block_delta", "index": c.blockIndex, "delta": map[string]interface{}{"type": "text_delta", "text": content}}); err != nil {
			return err
		}
	}
	if rawCalls, ok := choice.Delta["tool_calls"].([]interface{}); ok && len(rawCalls) > 0 {
		for _, rawCall := range rawCalls {
			call, ok := rawCall.(map[string]interface{})
			if !ok {
				continue
			}
			// OpenAI streams each parallel call under its own index; a call's
			// id/name arrive in its first delta and only argument fragments
			// follow. Missing index defaults to 0 for gateways that omit it.
			callIndex := 0
			if idx, ok := call["index"].(float64); ok {
				callIndex = int(idx)
			}
			function, _ := call["function"].(map[string]interface{})

			// The first call streams live as an Anthropic block so clients
			// watch its arguments fill in as they arrive. Any further call
			// with a different index is buffered and emitted sequentially at
			// finish, keeping the block lifecycle strictly ordered.
			if c.liveToolIndex == -1 || callIndex == c.liveToolIndex {
				if c.liveToolIndex == -1 {
					c.liveToolIndex = callIndex
					if c.blockType != "tool_use" {
						c.toolID, _ = call["id"].(string)
						c.toolName, _ = function["name"].(string)
						c.toolName = restoreToolName(c.aliases, c.toolName)
						index := 0
						if c.blockType != "" {
							index = c.blockIndex + 1
						}
						if err := c.startBlock("tool_use", index); err != nil {
							return err
						}
					}
				}
				arguments, _ := function["arguments"].(string)
				if arguments != "" {
					if err := c.event("content_block_delta", map[string]interface{}{"type": "content_block_delta", "index": c.blockIndex, "delta": map[string]interface{}{"type": "input_json_delta", "partial_json": arguments}}); err != nil {
						return err
					}
				}
				continue
			}

			buffer, seen := c.toolCalls[callIndex]
			if !seen {
				buffer = &toolCallBuffer{}
				c.toolCalls[callIndex] = buffer
			}
			if id, _ := call["id"].(string); id != "" {
				buffer.id = id
			}
			if name, _ := function["name"].(string); name != "" {
				buffer.name = name
			}
			if arguments, _ := function["arguments"].(string); arguments != "" {
				buffer.args.WriteString(arguments)
			}
		}
		c.stopReason = "tool_calls"
	}
	return nil
}

func (c *anthropicStreamConverter) finish() error {
	if c.done {
		return nil
	}
	if err := c.start(); err != nil {
		return err
	}
	c.done = true
	if c.blockType != "" {
		if err := c.event("content_block_stop", map[string]interface{}{"type": "content_block_stop", "index": c.blockIndex}); err != nil {
			return err
		}
	}
	// Emit any buffered parallel tool calls as complete, strictly sequential
	// blocks. Clients cannot act on a tool before message_stop, so nothing
	// observable is delayed versus an interleaved emission.
	if len(c.toolCalls) > 0 {
		indexes := make([]int, 0, len(c.toolCalls))
		for idx := range c.toolCalls {
			indexes = append(indexes, idx)
		}
		sort.Ints(indexes)
		// Blocks continue from the last emitted one; with no open block the
		// first tool call owns index 0.
		next := c.blockIndex + 1
		if c.blockType == "" {
			next = 0
		}
		for _, idx := range indexes {
			call := c.toolCalls[idx]
			name := restoreToolName(c.aliases, call.name)
			content := map[string]interface{}{"type": "tool_use", "id": call.id, "name": name, "input": map[string]interface{}{}}
			if err := c.event("content_block_start", map[string]interface{}{"type": "content_block_start", "index": next, "content_block": content}); err != nil {
				return err
			}
			if args := call.args.String(); args != "" {
				if err := c.event("content_block_delta", map[string]interface{}{"type": "content_block_delta", "index": next, "delta": map[string]interface{}{"type": "input_json_delta", "partial_json": args}}); err != nil {
					return err
				}
			}
			if err := c.event("content_block_stop", map[string]interface{}{"type": "content_block_stop", "index": next}); err != nil {
				return err
			}
			next++
		}
	}
	reason := c.stopReason
	switch reason {
	case "", "stop":
		reason = "end_turn"
	case "length":
		reason = "max_tokens"
	case "tool_calls", "function_call":
		reason = "tool_use"
	}
	if err := c.event("message_delta", map[string]interface{}{"type": "message_delta", "delta": map[string]interface{}{"stop_reason": reason, "stop_sequence": nil}, "usage": map[string]interface{}{"output_tokens": 0}}); err != nil {
		return err
	}
	return c.event("message_stop", map[string]interface{}{"type": "message_stop"})
}

func (c *anthropicStreamConverter) usage() parsedTokenUsage {
	if c == nil {
		return parsedTokenUsage{}
	}
	return c.tokenUsage
}

func openAIJSONToAnthropic(body []byte, model string) ([]byte, error) {
	return openAIJSONToAnthropicWithAliases(body, model, nil)
}

func openAIJSONToAnthropicWithAliases(body []byte, model string, aliases *toolNameAliases) ([]byte, error) {
	var response openAIChunk
	if err := json.Unmarshal(body, &response); err != nil {
		return nil, err
	}
	if response.Model != "" {
		model = response.Model
	}
	content := make([]interface{}, 0, 2)
	stopReason := "end_turn"
	if len(response.Choices) > 0 {
		choice := response.Choices[0]
		if reason := choice.FinishReason; reason != "" {
			switch reason {
			case "length":
				stopReason = "max_tokens"
			case "tool_calls", "function_call":
				stopReason = "tool_use"
			}
		}
		if reasoning, _ := choice.Message["reasoning_content"].(string); reasoning != "" {
			content = append(content, map[string]interface{}{"type": "thinking", "thinking": reasoning, "signature": ""})
		}
		if text, _ := choice.Message["content"].(string); text != "" {
			content = append(content, map[string]interface{}{"type": "text", "text": text})
		}
		if calls, ok := choice.Message["tool_calls"].([]interface{}); ok {
			for _, raw := range calls {
				call, _ := raw.(map[string]interface{})
				fn, _ := call["function"].(map[string]interface{})
				input := map[string]interface{}{}
				if arguments, ok := fn["arguments"].(string); ok && arguments != "" {
					_ = json.Unmarshal([]byte(arguments), &input)
				}
				name, _ := fn["name"].(string)
				content = append(content, map[string]interface{}{"type": "tool_use", "id": call["id"], "name": restoreToolName(aliases, name), "input": input})
			}
		}
	}
	usage := map[string]interface{}{"input_tokens": 0, "output_tokens": 0}
	if response.Usage != nil {
		if value, ok := response.Usage["prompt_tokens"]; ok {
			usage["input_tokens"] = value
		}
		if value, ok := response.Usage["completion_tokens"]; ok {
			usage["output_tokens"] = value
		}
	}
	out := map[string]interface{}{
		"id": response.ID, "type": "message", "role": "assistant", "model": model,
		"content": content, "stop_reason": stopReason, "stop_sequence": nil, "usage": usage,
	}
	return json.Marshal(out)
}

func restoreToolName(aliases *toolNameAliases, name string) string {
	if aliases == nil {
		return name
	}
	return aliases.restoreName(name)
}
