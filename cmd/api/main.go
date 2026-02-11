package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"time"

	openai "github.com/sashabaranov/go-openai"
)

// func runEvaluation() error {
// 	cmd := exec.Command("promptfoo", "eval")
// 	cmd.Stdout = os.Stdout
// 	cmd.Stderr = os.Stderr
// 	fmt.Println("Running Promptfoo evaluation...")
// 	return cmd.Run()
// }

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

	// tools := []openai.Tool{
	// 	{
	// 		Type: openai.ToolTypeFunction,
	// 		Function: &openai.FunctionDefinition{
	// 			Name:        "get_current_time",
	// 			Description: "Get the current time and date in Bangladesh",
	// 		},
	// 	},
	// }

	// api calls:
	stream, err := client.CreateChatCompletionStream(ctx, openai.ChatCompletionRequest{
		Model:     openai.GPT4oMini,
		MaxTokens: 1000,
		Messages:  messages,
		// Tools:     tools,
		Stream: true,
	})
	if err != nil {
		log.Fatalf("ChatCompletion error: %v", err)
	}

	defer stream.Close()

	for {
		response, err := stream.Recv()
		if err != nil {
			log.Fatalf("Stream error: %v", err)
		}

		if response.Choices[0].FinishReason != "" {
			break
		}

		fmt.Printf("%s", response.Choices[0].Delta.Content)

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
