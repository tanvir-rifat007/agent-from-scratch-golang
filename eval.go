package main

import "os/exec"

func main() {

	cmd := exec.Command("promptfoo", "eval")
	cmd.Run()
}
