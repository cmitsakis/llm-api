package conversation

import (
	"fmt"
	"testing"
)

func TestAppendTokenToLastMessageAssistant(t *testing.T) {
	c := NewConversation("{{ system_prompt }}")
	c.AddMessageUser("{{ user_msg_1 }}")
	c.AppendTokenToLastMessageAssistant("  \n")
	c.AppendTokenToLastMessageAssistant("\n  ")
	c.AppendTokenToLastMessageAssistant(" \n ")
	msg, err := c.LastMessageOfAssistant()
	if err != nil {
		fmt.Printf("c.LastMessageOfAssistant() failed: %s\n", err)
		t.Fail()
		return
	}
	if msg != "" {
		fmt.Printf("last message of assistant = %x\n", msg)
		t.Fail()
		return
	}
	c.AppendTokenToLastMessageAssistant("  \n  {{ assistant_msg_1 }}")
	msg, err = c.LastMessageOfAssistant()
	if err != nil {
		fmt.Printf("c.LastMessageOfAssistant() failed: %s\n", err)
		t.Fail()
		return
	}
	if msg != "{{ assistant_msg_1 }}" {
		fmt.Printf("last message of assistant = %x\n", msg)
		t.Fail()
		return
	}
}

func TestGeneratePrompt(t *testing.T) {
	c := NewConversation("{{ system_prompt }}")

	// 1 user message
	c.AddMessageUser("{{ user_msg_1 }}")
	testPrompt(t, c, PromptTemplateLlama2, `<s>[INST] <<SYS>>
{{ system_prompt }}
<</SYS>>

{{ user_msg_1 }} [/INST]`)
	testPrompt(t, c, PromptTemplateVicunaV11, `A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {{ user_msg_1 }} ASSISTANT:`)

	// 2 user messages
	c.AddMessageAssistant("{{ assistant_msg_1 }}")
	c.AddMessageUser("{{ user_msg_2 }}")
	testPrompt(t, c, PromptTemplateLlama2, `<s>[INST] <<SYS>>
{{ system_prompt }}
<</SYS>>

{{ user_msg_1 }} [/INST] {{ assistant_msg_1 }} </s><s>[INST] {{ user_msg_2 }} [/INST]`)
	testPrompt(t, c, PromptTemplateVicunaV11, `A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {{ user_msg_1 }} ASSISTANT: {{ assistant_msg_1 }}</s>USER: {{ user_msg_2 }} ASSISTANT:`)

	// 3 user messages
	c.AddMessageAssistant("{{ assistant_msg_2 }}")
	c.AddMessageUser("{{ user_msg_3 }}")
	testPrompt(t, c, PromptTemplateLlama2, `<s>[INST] <<SYS>>
{{ system_prompt }}
<</SYS>>

{{ user_msg_1 }} [/INST] {{ assistant_msg_1 }} </s><s>[INST] {{ user_msg_2 }} [/INST] {{ assistant_msg_2 }} </s><s>[INST] {{ user_msg_3 }} [/INST]`)
	testPrompt(t, c, PromptTemplateVicunaV11, `A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {{ user_msg_1 }} ASSISTANT: {{ assistant_msg_1 }}</s>USER: {{ user_msg_2 }} ASSISTANT: {{ assistant_msg_2 }}</s>USER: {{ user_msg_3 }} ASSISTANT:`)
}

func testPrompt(t *testing.T, c Conversation, promptTemplate PromptTemplate, expectedPrompt string) {
	t.Helper()
	prompt, err := c.GeneratePrompt(promptTemplate)
	if err != nil {
		fmt.Printf("GeneratePrompt() failed: %s\n", err)
		t.Fail()
		return
	}
	if prompt != expectedPrompt {
		fmt.Printf("Generated prompt differs from expected:\nGenerated Prompt:\n%v\nExpected Prompt:\n%v\n", prompt, expectedPrompt)
		t.Fail()
	}
}
