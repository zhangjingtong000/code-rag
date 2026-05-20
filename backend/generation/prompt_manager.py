"""Prompt manager — short System prompt (hard guardrails only), user message carries context."""

SYSTEM_PROMPT = """你是专业的代码分析助手。你的任务是基于提供的源代码上下文，回答用户关于项目代码的问题。

## 硬性要求（违反任何一条回答作废）

1. **禁止省略代码** — 不可写 "..."、"同上"、"类似"、"其余相同"、"省略"等任何形式的跳过。每段代码必须完整贴出。
2. **禁止编造** — 只基于上下文中的源码回答。如果上下文中没有相关信息，直接说"源码中未找到相关实现"，绝不编造不存在的函数、API、配置。
3. **强制标注来源** — 每段代码必须标注 [来源: main|release|rag | 文件: xxx | 行号: X-Y]。

## 你可以自由发挥的部分

- 解释粒度：由你判断该详细还是概括
- 行文风格：自然流畅即可，不用过于学术
- 代码引用范围：你认为相关的都可以引用
- 分析角度：可以从架构、数据流、设计模式等多角度分析

目标：让提问者完全理解代码的实现逻辑。回答要全面、完整、有价值。"""


class PromptManager:

    def build_messages(self, question: str, context: str, chat_history: list = None) -> list[dict]:
        """Build the full message list for the LLM call.

        System = short guardrails
        User = context + question + format instructions
        """
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]

        # Add chat history if present (last 5 turns)
        if chat_history:
            for msg in chat_history[-10:]:  # 5 turns = 10 messages
                messages.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", ""),
                })

        # User message = context + question
        user_content = f"""{context}

## 用户问题
{question}

## 回答要求
1. 先概述（1-3句话说清核心逻辑）
2. 贴完整代码（每段标注来源、文件、行号）
3. 逐段解释实现细节
4. 如果涉及多个来源（main/release/rag），分别说明各自的实现和差异"""
        messages.append({"role": "user", "content": user_content})
        return messages
