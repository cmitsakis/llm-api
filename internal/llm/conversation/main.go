package conversation

import (
	"bytes"
	"errors"
	"fmt"
	"strings"
	"text/template"
	"unicode"
)

type Role string

const (
	RoleSystem    Role = "system"
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
)

type Message struct {
	Role Role
	Text string
}

type Conversation struct {
	SystemPrompt string
	Messages     []Message
}

func NewConversation(systemPrompt string) Conversation {
	var c Conversation
	if systemPrompt != "" {
		c.SetSystemPrompt(systemPrompt)
	}
	return c
}

var (
	errNoMessagesInConversation          = errors.New("no messages in conversation")
	errNoUserMessagesInConversation      = errors.New("no user messages in conversation")
	errNoAssistantMessagesInConversation = errors.New("no assistant messages in conversation")
)

// returns all the messages except for the system prompt.
func (c Conversation) MessagesWithoutSystemPrompt() []Message {
	if len(c.Messages) == 0 {
		return nil
	}
	if c.Messages[0].Role != RoleSystem {
		// there is no system prompt
		return c.Messages
	}
	// there is system prompt
	if len(c.Messages) == 1 {
		// there is system prompt, but no other messages
		return nil
	}
	return c.Messages[1:]
}

// returns the last message with role=user.
// assumes no two messages in a row can have the same role.
func (c Conversation) LastMessageOfUser() (string, error) {
	lastIndex := len(c.Messages) - 1
	if lastIndex < 0 {
		return "", errNoMessagesInConversation
	}
	if c.Messages[lastIndex].Role == RoleUser {
		return c.Messages[lastIndex].Text, nil
	}
	lastIndex--
	if lastIndex < 0 {
		return "", errNoUserMessagesInConversation
	}
	if c.Messages[lastIndex].Role == RoleUser {
		return c.Messages[lastIndex].Text, nil
	}
	return "", errNoUserMessagesInConversation
}

// returns the last message with role=assistant.
// assumes no two messages in a row can have the same role.
func (c Conversation) LastMessageOfAssistant() (string, error) {
	lastIndex := len(c.Messages) - 1
	if lastIndex < 0 {
		return "", errNoMessagesInConversation
	}
	if c.Messages[lastIndex].Role == RoleAssistant {
		return c.Messages[lastIndex].Text, nil
	}
	lastIndex--
	if lastIndex < 0 {
		return "", errNoAssistantMessagesInConversation
	}
	if c.Messages[lastIndex].Role == RoleAssistant {
		return c.Messages[lastIndex].Text, nil
	}
	return "", errNoAssistantMessagesInConversation
}

func (c *Conversation) SetSystemPrompt(text string) {
	c.SystemPrompt = text
	msg := Message{Role: RoleSystem, Text: text}
	if len(c.Messages) == 0 {
		c.Messages = []Message{msg}
	} else {
		c.Messages[0] = msg
	}
}

func (c *Conversation) AddMessageUser(text string) {
	lastIndex := len(c.Messages) - 1
	if lastIndex >= 0 && c.Messages[lastIndex].Role == RoleUser {
		c.Messages[lastIndex].Text = c.Messages[lastIndex].Text + "\n" + text
		return
	}
	c.Messages = append(c.Messages, Message{Role: RoleUser, Text: text})
}

func (c *Conversation) AddMessageAssistant(text string) {
	lastIndex := len(c.Messages) - 1
	if lastIndex >= 0 && c.Messages[lastIndex].Role == RoleAssistant {
		c.Messages[lastIndex].Text = c.Messages[lastIndex].Text + "\n" + text
		return
	}
	c.Messages = append(c.Messages, Message{Role: RoleAssistant, Text: text})
}

func (c *Conversation) SetLastMessageOfAssistant(text string) {
	lastIndex := len(c.Messages) - 1
	if lastIndex >= 0 && c.Messages[lastIndex].Role == RoleAssistant {
		c.Messages[lastIndex].Text = text
		return
	}
	c.Messages = append(c.Messages, Message{Role: RoleAssistant, Text: text})
}

// left-trims space from the token if str is the empty string, and appends the possibly trimmed token to str.
// This function typically is called repeatedly for several tokens on the same str string that accumulates the tokens.
// The purpose of this function is to make sure str has no leading space characters.
func TrimAndAppend(str string, token string) (string, string) {
	if str == "" {
		// this is the first token to be appended to str that might contain non-space characters
		token = strings.TrimLeftFunc(token, unicode.IsSpace)
	}
	return str + token, token
}

func (c *Conversation) AppendTokenToLastMessageAssistant(token string) string {
	lastIndex := len(c.Messages) - 1
	if lastIndex >= 0 && c.Messages[lastIndex].Role == RoleAssistant {
		c.Messages[lastIndex].Text, token = TrimAndAppend(c.Messages[lastIndex].Text, token)
		return token
	}
	// this token is the first token of the assistant message
	tokenTrimmed := strings.TrimLeftFunc(token, unicode.IsSpace)
	c.Messages = append(c.Messages, Message{Role: RoleAssistant, Text: tokenTrimmed})
	return tokenTrimmed
}

type PromptTemplate struct {
	*template.Template
	RequiresSystemPrompt bool
}

func NewPromptTemplate(promptTemplateString string) (PromptTemplate, error) {
	t, err := template.New("promptTemplate").Parse(promptTemplateString)
	if err != nil {
		return PromptTemplate{}, fmt.Errorf("failed to parse prompt template: %w", err)
	}
	return PromptTemplate{t, false}, nil
}

const promptTemplateStringLlama2 = `
{{define "prompt" -}}
<s>{{range $i, $m := .Messages}}{{if eq $m.Role "system"}}[INST] <<SYS>>
{{$m.Text}}
<</SYS>>

{{else if eq $i 1 -}}
{{$m.Text}} [/INST]
{{- else if eq $m.Role "assistant" }} {{$m.Text}} </s><s>
{{- else if eq $m.Role "user" -}}
[INST] {{$m.Text}} [/INST]
{{- end}}
{{- end}}
{{- end}}
`

var PromptTemplateLlama2 = PromptTemplate{template.Must(template.New("llama-2").Parse(promptTemplateStringLlama2)), true}

const promptTemplateStringVicunaV11 = `
{{define "prompt"}}A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. {{range $i, $m := .MessagesWithoutSystemPrompt}}{{if eq $m.Role "user" }}{{if gt $i 1 }}</s>{{end}}USER: {{$m.Text}}{{else if eq $m.Role "assistant" }} ASSISTANT: {{$m.Text}}{{end}}{{end}} ASSISTANT:{{end}}
`

var PromptTemplateVicunaV11 = PromptTemplate{template.Must(template.New("vicuna_v1.1").Parse(promptTemplateStringVicunaV11)), false}

func (c Conversation) GeneratePrompt(promptTemplate PromptTemplate) (string, error) {
	buf := &bytes.Buffer{}
	err := promptTemplate.ExecuteTemplate(buf, "prompt", c)
	if err != nil {
		return "", fmt.Errorf("failed to execute prompt template: %w", err)
	}
	return buf.String(), nil
}
