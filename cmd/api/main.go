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

func main() {
	log.SetFlags(0)

	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		log.Fatal("OPENAI_API_KEY not set")
	}

	client := openai.NewClient(apiKey)
	ctx := context.Background()

	// Conversation memory
	messages := []openai.ChatCompletionMessage{
		{
			Role:    openai.ChatMessageRoleSystem,
			Content: "You are a helpful assistant. If user asks for Bangladesh time, use the tool.",
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

		// Add user message
		messages = append(messages, openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleUser,
			Content: userInput,
		})

		// First stream
		stream, err := client.CreateChatCompletionStream(ctx, openai.ChatCompletionRequest{
			Model:    openai.GPT4oMini,
			Messages: messages,
			Tools:    tools,
			Stream:   true,
		})
		if err != nil {
			log.Fatal(err)
		}

		defer stream.Close()

		// Collect assistant message and tool calls
		var assistantMessage openai.ChatCompletionMessage
		assistantMessage.Role = openai.ChatMessageRoleAssistant
		toolCalls := make(map[int]*openai.ToolCall)

		fmt.Print("Assistant: ")

		// this loop is for streaming
		for {
			response, err := stream.Recv()
			if err != nil {
				break
			}

			delta := response.Choices[0].Delta

			// Stream content
			if delta.Content != "" {
				fmt.Print(delta.Content)
				assistantMessage.Content += delta.Content
			}

			// Collect tool calls
			for _, tc := range delta.ToolCalls {
				fmt.Printf("%+v\n", tc)
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

		// Attach tool calls if any
		for _, tc := range toolCalls {
			assistantMessage.ToolCalls = append(assistantMessage.ToolCalls, *tc)
		}

		// Save assistant message
		messages = append(messages, assistantMessage)

		// HANDLE TOOL CALLS
		if len(assistantMessage.ToolCalls) > 0 {

			fmt.Println("\n🔧 Executing tool...")

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

			// final stream to get assistant's response after tool execution

			finalStream, err := client.CreateChatCompletionStream(ctx, openai.ChatCompletionRequest{
				Model:    openai.GPT4oMini,
				Messages: messages,
				Stream:   true,
			})
			if err != nil {
				log.Fatal(err)
			}

			var finalAssistant openai.ChatCompletionMessage
			finalAssistant.Role = openai.ChatMessageRoleAssistant

			fmt.Print("Assistant: ")

			for {
				resp, err := finalStream.Recv()
				if err != nil {
					break
				}

				content := resp.Choices[0].Delta.Content
				fmt.Print(content)
				finalAssistant.Content += content
			}

			finalStream.Close()

			messages = append(messages, finalAssistant)
		}

		fmt.Println("\n")
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

	resultJSON, _ := json.Marshal(result)
	return string(resultJSON)
}

func primeMinisterOfBangladesh() string {
	return "Dr. Yunus is the prime minister of Bangladesh."
}
