package main

import (
	"context"
	"fmt"
	"strings"

	"github.com/pkoukk/tiktoken-go"
	openai "github.com/sashabaranov/go-openai"
)

type AgentMemory struct {
	SystemMessages []openai.ChatCompletionMessage
	Summary        string
	RecentTurns    [][]openai.ChatCompletionMessage
}

const (
	MODEL_CONTEXT_LIMIT = 4000
	COMPLETION_BUFFER   = 2000
	SAFETY_MARGIN       = 1000
	SUMMARY_MAX_TOKENS  = 800
)

func AllowedHistoryTokens() int {
	return MODEL_CONTEXT_LIMIT - COMPLETION_BUFFER - SAFETY_MARGIN
}

func AddTurn(mem *AgentMemory, turn []openai.ChatCompletionMessage) {
	mem.RecentTurns = append(mem.RecentTurns, turn)
}

func BuildContext(mem *AgentMemory) []openai.ChatCompletionMessage {
	var msgs []openai.ChatCompletionMessage

	msgs = append(msgs, mem.SystemMessages...)

	if mem.Summary != "" {
		msgs = append(msgs, openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleSystem,
			Content: "Conversation Summary:\n" + mem.Summary,
		})
	}

	for _, t := range mem.RecentTurns {
		msgs = append(msgs, t...)
	}

	return msgs
}

func countMessagesTokens(messages []openai.ChatCompletionMessage, model string) int {
	enc, err := tiktoken.EncodingForModel(model)
	if err != nil {
		// fallback to rough estimate
		total := 0
		for _, m := range messages {
			total += len(m.Content) / 4
		}
		return total
	}

	total := 0
	for _, m := range messages {
		// ~4 tokens overhead per message (role, separators)
		total += 4
		total += len(enc.Encode(m.Content, nil, nil))

		// count tool call arguments too
		for _, tc := range m.ToolCalls {
			total += len(enc.Encode(tc.Function.Name, nil, nil))
			total += len(enc.Encode(tc.Function.Arguments, nil, nil))
		}
	}
	return total
}

func CompactIfNeeded(
	ctx context.Context,
	client *openai.Client,
	model string,
	mem *AgentMemory,
) error {

	tokenBudget := AllowedHistoryTokens()

	current := BuildContext(mem)

	if countMessagesTokens(current, model) <= tokenBudget {
		return nil
	}

	if len(mem.RecentTurns) < 2 {
		return nil
	}

	// summarize oldest 2 turns
	oldTurns := mem.RecentTurns[:2]

	summary, err := SummarizeTurns(ctx, client, model, oldTurns)
	if err != nil {
		return err
	}

	if mem.Summary == "" {
		mem.Summary = summary
	} else {
		mem.Summary = mergeSummaries(ctx, client, model, mem.Summary, summary)
	}

	mem.RecentTurns = mem.RecentTurns[2:]

	fmt.Println("🧠 Memory compacted via summarization")

	return nil
}

func SummarizeTurns(
	ctx context.Context,
	client *openai.Client,
	model string,
	turns [][]openai.ChatCompletionMessage,
) (string, error) {

	var convo strings.Builder

	for _, turn := range turns {
		for _, msg := range turn {
			if msg.Content != "" {
				convo.WriteString(fmt.Sprintf("%s: %s\n", msg.Role, msg.Content))
			}
		}
	}

	req := openai.ChatCompletionRequest{
		Model: model,
		Messages: []openai.ChatCompletionMessage{
			{
				Role:    openai.ChatMessageRoleSystem,
				Content: "You are a memory compression engine. Summarize while preserving important facts, decisions, and user preferences.",
			},
			{
				Role:    openai.ChatMessageRoleUser,
				Content: convo.String(),
			},
		},
		MaxTokens: SUMMARY_MAX_TOKENS,
	}

	resp, err := client.CreateChatCompletion(ctx, req)
	if err != nil {
		return "", err
	}

	return resp.Choices[0].Message.Content, nil
}

func mergeSummaries(
	ctx context.Context,
	client *openai.Client,
	model string,
	existing string,
	newSummary string,
) string {

	req := openai.ChatCompletionRequest{
		Model: model,
		Messages: []openai.ChatCompletionMessage{
			{
				Role:    openai.ChatMessageRoleSystem,
				Content: "Merge the following conversation summaries into one concise but complete summary.",
			},
			{
				Role: openai.ChatMessageRoleUser,
				Content: fmt.Sprintf("Existing Summary:\n%s\n\nNew Summary:\n%s",
					existing, newSummary),
			},
		},
		MaxTokens: SUMMARY_MAX_TOKENS,
	}

	resp, err := client.CreateChatCompletion(ctx, req)
	if err != nil {
		return existing + "\n" + newSummary
	}

	return resp.Choices[0].Message.Content
}
