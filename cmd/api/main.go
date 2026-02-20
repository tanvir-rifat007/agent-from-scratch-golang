//

// system prompt -> user message -> assistant message (with tool calls) -> tool response -> assistant final toolResponse

//

package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"strings"
	"time"

	openai "github.com/sashabaranov/go-openai"
)

const MAX_TURNS = 10

func main() {
	log.SetFlags(0)

	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		log.Fatal("OPENAI_API_KEY not set")
	}

	client := openai.NewClient(apiKey)
	ctx := context.Background()

	messages := []openai.ChatCompletionMessage{
		{
			Role:    openai.ChatMessageRoleSystem,
			Content: "You are a helpful assistant. Use tools whenever relevant.",
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

	// this is for the promptfoo evaluation
	if len(os.Args) > 1 {

		userInput := strings.Join(os.Args[1:], " ")
		messages = append(messages, openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleUser,
			Content: userInput,
		})
		runAgentLoop(ctx, client, &messages, tools)
		return

	}

	reader := bufio.NewReader(os.Stdin)
	fmt.Println("Chat started (type 'exit' to quit)\n")

	for {
		fmt.Print("You: ")
		userInput, _ := reader.ReadString('\n')
		userInput = strings.TrimSpace(userInput)

		if userInput == "exit" {
			fmt.Println("Goodbye 👋")
			break
		}

		messages = append(messages, openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleUser,
			Content: userInput,
		})

		// this runAgent means: Agent in loop
		runAgentLoop(ctx, client, &messages, tools)
		fmt.Println()
		fmt.Println()

	}
}

func runAgentLoop(ctx context.Context, client *openai.Client, messages *[]openai.ChatCompletionMessage, tools []openai.Tool) {
	iteration := 0
	for {

		if iteration >= MAX_TURNS {
			fmt.Println("\nMaximum turns reached. Ending chat.")
			break
		}
		iteration++

		stream, err := client.CreateChatCompletionStream(ctx, openai.ChatCompletionRequest{
			Model:    openai.GPT4oMini,
			Messages: *messages,
			Tools:    tools,
			Stream:   true,
		})
		if err != nil {
			log.Fatal(err)
		}

		var assistantMsg openai.ChatCompletionMessage
		assistantMsg.Role = openai.ChatMessageRoleAssistant
		toolCalls := make(map[int]*openai.ToolCall)

		for {
			response, err := stream.Recv()
			if err != nil {
				break
			}
			delta := response.Choices[0].Delta
			if delta.Content != "" {
				fmt.Print(delta.Content)
				assistantMsg.Content += delta.Content
			}
			for _, tc := range delta.ToolCalls {
				if _, exists := toolCalls[*tc.Index]; !exists {
					toolCalls[*tc.Index] = &openai.ToolCall{
						ID:       tc.ID,
						Type:     tc.Type,
						Function: openai.FunctionCall{Name: tc.Function.Name},
					}
				}
				toolCalls[*tc.Index].Function.Arguments += tc.Function.Arguments
			}
		}
		stream.Close()

		for i := 0; i < len(toolCalls); i++ {
			if tc, ok := toolCalls[i]; ok {
				assistantMsg.ToolCalls = append(assistantMsg.ToolCalls, *tc)
			}
		}
		*messages = append(*messages, assistantMsg)

		if len(assistantMsg.ToolCalls) == 0 {
			break // model is done
		}

		for _, tc := range assistantMsg.ToolCalls {
			var result string
			switch tc.Function.Name {
			case "get_current_time":
				result = getCurrentDateTime()
			case "prime_minister_of_bangladesh":
				result = primeMinisterOfBangladesh()
			default:
				result = `{"error": "unknown tool"}`
			}
			*messages = append(*messages, openai.ChatCompletionMessage{
				Role:       openai.ChatMessageRoleTool,
				Content:    result,
				ToolCallID: tc.ID,
			})
		}
	}
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

	data, _ := json.Marshal(result)
	return string(data)
}

func primeMinisterOfBangladesh() string {
	return `{"prime_minister": "Dr. Muhammad Yunus"}`
}
