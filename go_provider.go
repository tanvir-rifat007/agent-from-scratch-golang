package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"time"

	openai "github.com/sashabaranov/go-openai"
)

func runAgent(userInput string) (string, error) {
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		return "", fmt.Errorf("OPENAI_API_KEY not set")
	}

	client := openai.NewClient(apiKey)
	ctx := context.Background()

	messages := []openai.ChatCompletionMessage{
		{
			Role:    openai.ChatMessageRoleSystem,
			Content: "You are a helpful assistant. If user asks for Bangladesh time and prime minister of bangladesh, use tools",
		},
		{
			Role:    openai.ChatMessageRoleUser,
			Content: userInput,
		},
	}

	tools := []openai.Tool{
		{
			Type: openai.ToolTypeFunction,
			Function: &openai.FunctionDefinition{
				Name:        "get_current_time",
				Description: "Get the current time and date in Bangladesh",
			},
		},
		{
			Type: openai.ToolTypeFunction,
			Function: &openai.FunctionDefinition{
				Name:        "prime_minister_of_bangladesh",
				Description: "Get the name of the current prime minister of Bangladesh",
			},
		},
	}

	// First call
	stream, err := client.CreateChatCompletionStream(ctx, openai.ChatCompletionRequest{
		Model:    openai.GPT4oMini,
		Messages: messages,
		Tools:    tools,
	})
	if err != nil {
		return "", err
	}

	// Collect assistant message and tool calls
	var assistantMessage openai.ChatCompletionMessage
	assistantMessage.Role = openai.ChatMessageRoleAssistant
	toolCalls := make(map[int]*openai.ToolCall)

	var fullResponse strings.Builder

	for {
		response, err := stream.Recv()
		if err != nil {
			break
		}

		delta := response.Choices[0].Delta

		if delta.Content != "" {
			fullResponse.WriteString(delta.Content)
			assistantMessage.Content += delta.Content
		}

		for _, tc := range delta.ToolCalls {
			if _, exists := toolCalls[*tc.Index]; !exists {
				toolCalls[*tc.Index] = &openai.ToolCall{
					ID:   tc.ID,
					Type: tc.Type,
					Function: openai.FunctionCall{
						Name:      tc.Function.Name,
						Arguments: "",
					},
				}
			}
			toolCalls[*tc.Index].Function.Arguments += tc.Function.Arguments
		}
	}
	stream.Close()

	// Attach tool calls if any
	for _, tc := range toolCalls {
		assistantMessage.ToolCalls = append(assistantMessage.ToolCalls, *tc)
	}

	messages = append(messages, assistantMessage)

	// HANDLE TOOL CALLS
	if len(assistantMessage.ToolCalls) > 0 {
		for _, tc := range assistantMessage.ToolCalls {
			var toolResponse string

			switch tc.Function.Name {
			case "get_current_time":
				toolResponse = getCurrentDateTime()
			case "prime_minister_of_bangladesh":
				toolResponse = primeMinisterOfBangladesh()
			default:
				toolResponse = `{"error": "unknown tool"}`
			}

			messages = append(messages, openai.ChatCompletionMessage{
				Role:       openai.ChatMessageRoleTool,
				Content:    toolResponse,
				ToolCallID: tc.ID,
			})
		}

		// Final stream
		finalStream, err := client.CreateChatCompletionStream(ctx, openai.ChatCompletionRequest{
			Model:    openai.GPT4oMini,
			Messages: messages,
		})
		if err != nil {
			return "", err
		}

		var finalAssistant openai.ChatCompletionMessage
		finalAssistant.Role = openai.ChatMessageRoleAssistant

		fullResponse.Reset() // Clear previous content

		for {
			resp, err := finalStream.Recv()
			if err != nil {
				break
			}

			content := resp.Choices[0].Delta.Content
			fullResponse.WriteString(content)
			finalAssistant.Content += content
		}
		finalStream.Close()
	}

	return fullResponse.String(), nil
}

func getCurrentDateTime() string {
	loc, err := time.LoadLocation("Asia/Dhaka")
	if err != nil {
		loc = time.UTC
	}

	now := time.Now().In(loc)

	result := map[string]any{
		"datetime":  now.Format(time.RFC3339),
		"date":      now.Format("2006-01-02"),
		"time":      now.Format("15:04:05"),
		"timezone":  "Asia/Dhaka",
		"timestamp": now.Unix(),
	}

	resultJSON, _ := json.Marshal(result)
	return string(resultJSON)
}

func primeMinisterOfBangladesh() string {
	return `{"name": "Dr. Yunus", "title": "Prime Minister"}`
}

func main() {
	// Check if input comes from args or stdin
	var userInput string

	if len(os.Args) > 1 {
		// Promptfoo passes prompt as first argument
		userInput = strings.TrimSpace(os.Args[1])
	} else {
		// Fallback to stdin for manual testing
		scanner := bufio.NewScanner(os.Stdin)
		if !scanner.Scan() {
			os.Exit(1)
		}
		userInput = strings.TrimSpace(scanner.Text())
	}

	// Run the agent
	output, err := runAgent(userInput)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}

	// Print only the output (Promptfoo expects clean stdout)
	fmt.Print(output)
}
