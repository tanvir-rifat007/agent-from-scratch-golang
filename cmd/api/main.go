package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"os/exec"
	"time"

	openai "github.com/sashabaranov/go-openai"
)

func runEvaluation() error {
	cmd := exec.Command("promptfoo", "eval")
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	fmt.Println("Running Promptfoo evaluation...")
	return cmd.Run()
}

func main() {
	log.SetFlags(0)
	client := openai.NewClient(os.Getenv("OPENAI_API_KEY"))
	ctx := context.Background()
	messages := []openai.ChatCompletionMessage{
		{
			Role:    openai.ChatMessageRoleSystem,
			Content: "You are a helpful assistant that provides the current time and date in Bangladesh when asked.",
		},
		{
			Role:    openai.ChatMessageRoleUser,
			Content: "What is the time and date now in Bangladesh?",
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
	}

	// api calls:
	resp, err := client.CreateChatCompletion(ctx, openai.ChatCompletionRequest{
		Model:    openai.GPT4oMini,
		Messages: messages,
		Tools:    tools,
	})
	if err != nil {
		log.Fatalf("ChatCompletion error: %v", err)
	}

	message := resp.Choices[0].Message
	messages = append(messages, message)

	if len(message.ToolCalls) > 0 {
		for _, toolCall := range message.ToolCalls {
			fmt.Printf("Tool Call : %s\n", toolCall.Function.Name)

			var result string
			if toolCall.Function.Name == "get_current_time" {
				result = getCurrentDateTime()
			}

			messages = append(messages, openai.ChatCompletionMessage{
				Role:       openai.ChatMessageRoleTool,
				Content:    result,
				ToolCallID: toolCall.ID,
			})
		}

		// MOVE THIS OUTSIDE THE LOOP - only call once after all tools
		resp, err = client.CreateChatCompletion(ctx, openai.ChatCompletionRequest{
			Model:    openai.GPT4oMini,
			Messages: messages,
			Tools:    tools,
		})
		if err != nil {
			fmt.Printf("chatCompletionError: %v\n", err)
			return
		}

		// Print the final response from tool call
		fmt.Println("\nAssistant:", resp.Choices[0].Message.Content)
	} else {
		// If no tool calls, print the assistant's response directly
		fmt.Println("\nAssistant:", message.Content)
	}

	if err := runEvaluation(); err != nil {
		log.Printf("Evaluation failed: %v", err)
	}
}

func getCurrentDateTime() string {
	now := time.Now()
	result := map[string]any{
		"datetime":  now.Format(time.RFC3339),
		"date":      now.Format("2006-01-02"),
		"time":      now.Format("15:04:05"),
		"timezone":  now.Location().String(),
		"timestamp": now.Unix(),
	}
	resultJSON, _ := json.Marshal(result)
	return string(resultJSON)
}
