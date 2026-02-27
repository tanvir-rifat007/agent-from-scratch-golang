//

// system prompt -> user message -> assistant message (with tool calls) -> tool response -> assistant final toolResponse

//

package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strings"
	"time"

	openai "github.com/sashabaranov/go-openai"
	"github.com/sashabaranov/go-openai/jsonschema"
)

const (
	MODEL     = openai.GPT4oMini
	MAX_TURNS = 10
)

func main() {
	log.SetFlags(0)

	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		log.Fatal("OPENAI_API_KEY not set")
	}

	client := openai.NewClient(apiKey)
	ctx := context.Background()

	//Initialize long-lived memory
	mem := &AgentMemory{
		SystemMessages: []openai.ChatCompletionMessage{
			{
				Role:    openai.ChatMessageRoleSystem,
				Content: "You are a helpful assistant. Use tools whenever relevant.",
			},
		},
	}

	tools := buildTools()

	// this is for promptfoo evaluation testing
	if len(os.Args) > 1 {

		userInput := strings.Join(os.Args[1:], " ")
		userMsg := openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleUser,
			Content: userInput,
		}

		runAgentLoop(ctx, client, mem, userMsg, tools, true)
		return

	}

	reader := bufio.NewReader(os.Stdin)
	fmt.Println("Chat started (type 'exit' to quit)")

	for {
		fmt.Print("You: ")
		userInput, _ := reader.ReadString('\n')
		userInput = strings.TrimSpace(userInput)

		if userInput == "exit" {
			fmt.Println("Goodbye 👋")
			break
		}

		userMsg := openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleUser,
			Content: userInput,
		}

		runAgentLoop(ctx, client, mem, userMsg, tools, true)
		fmt.Println()
	}
}

func runAgentLoop(
	ctx context.Context,
	client *openai.Client,
	mem *AgentMemory,
	userMsg openai.ChatCompletionMessage,
	tools []openai.Tool,
	verbose bool,
) {

	iteration := 0
	var turn []openai.ChatCompletionMessage
	turn = append(turn, userMsg)

	for {
		if iteration >= MAX_TURNS {
			fmt.Println("\nMaximum turns reached.")
			break
		}
		iteration++

		err := CompactIfNeeded(ctx, client, MODEL, mem)
		if err != nil {
			log.Println("memory compaction error:", err)
		}

		sendMessages := BuildContext(mem)
		sendMessages = append(sendMessages, turn...)

		stream, err := client.CreateChatCompletionStream(ctx, openai.ChatCompletionRequest{
			Model:    MODEL,
			Messages: sendMessages,
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
			if err == io.EOF {
				break
			}
			if err != nil {
				log.Println("stream error:", err)
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

		turn = append(turn, assistantMsg)

		if len(assistantMsg.ToolCalls) == 0 {
			break
		}

		if verbose {
			fmt.Println("\n\n🔧 Tool calls:")
			for _, tc := range assistantMsg.ToolCalls {
				fmt.Printf("  - %s(%s)\n", tc.Function.Name, tc.Function.Arguments)
			}
		}

		for _, tc := range assistantMsg.ToolCalls {

			result := executeTool(tc, verbose)

			toolMsg := openai.ChatCompletionMessage{
				Role:       openai.ChatMessageRoleTool,
				Content:    result,
				ToolCallID: tc.ID,
			}

			turn = append(turn, toolMsg)
		}
	}

	AddTurn(mem, turn)
}

func buildTools() []openai.Tool {
	return []openai.Tool{
		{
			Type: openai.ToolTypeFunction,
			Function: &openai.FunctionDefinition{
				Name:        "get_current_time",
				Description: "Get current date and time in Bangladesh",
			},
		},
		{
			Type: openai.ToolTypeFunction,
			Function: &openai.FunctionDefinition{
				Name:        "web_search",
				Description: "Search the web for real-time information",
				Parameters: jsonschema.Definition{
					Type: jsonschema.Object,
					Properties: map[string]jsonschema.Definition{
						"query": {Type: jsonschema.String},
					},
					Required: []string{"query"},
				},
			},
		},
	}
}

func executeTool(tc openai.ToolCall, verbose bool) string {

	switch tc.Function.Name {

	case "get_current_time":
		return getCurrentDateTime()

	case "web_search":
		var args struct {
			Query string `json:"query"`
		}
		if err := json.Unmarshal([]byte(tc.Function.Arguments), &args); err != nil {
			return `{"error":"invalid arguments"}`
		}
		if verbose {
			fmt.Printf("\n🌐 Searching: %s\n", args.Query)
		}
		return tavilySearch(args.Query)

	default:
		return `{"error":"unknown tool"}`
	}
}

func getCurrentDateTime() string {
	loc, _ := time.LoadLocation("Asia/Dhaka")
	now := time.Now().In(loc)
	result := map[string]any{
		"datetime": now.Format(time.RFC3339),
	}
	data, _ := json.Marshal(result)
	return string(data)
}

func tavilySearch(query string) string {
	apiKey := os.Getenv("TAVILY_API_KEY")
	if apiKey == "" {
		return `{"error":"TAVILY_API_KEY not set"}`
	}

	payload, _ := json.Marshal(map[string]any{
		"api_key":     apiKey,
		"query":       query,
		"max_results": 3,
	})

	resp, err := http.Post("https://api.tavily.com/search", "application/json", bytes.NewBuffer(payload))
	if err != nil {
		return `{"error":"search failed"}`
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)

	if len(body) > 5000 {
		body = body[:5000]
	}

	return string(body)
}
