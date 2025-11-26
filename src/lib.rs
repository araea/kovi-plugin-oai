//! kovi-plugin-openai-chat
//!
//! 符号指令系统 AI 聊天插件
//!
//! 指令格式: [&]["]智能体名[操作符][参数]
//!
//! 模式前缀: & 私有 | " 文本
//! 操作符: # 创建 | ~ 复制/重新 | / 查看 | - 删除 | _ 导出 | ' 编辑 | ! 停止
//! 对象符: @ 智能体 | $ 提示词 | % 模型 | : 描述
//! 范围符: * 全部 | 数字索引

// --- 类型定义 ---
mod types {
    use serde::{Deserialize, Serialize};
    use std::collections::{HashMap, HashSet};

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ChatMessage {
        pub role: String,
        pub content: String,
        #[serde(default)]
        pub images: Vec<String>,
        #[serde(default)]
        pub timestamp: i64,
    }

    impl ChatMessage {
        pub fn new(role: &str, content: &str, images: Vec<String>) -> Self {
            Self {
                role: role.to_string(),
                content: content.to_string(),
                images,
                timestamp: chrono::Local::now().timestamp(),
            }
        }
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct Agent {
        pub name: String,
        #[serde(default)]
        pub description: String,
        pub model: String,
        pub system_prompt: String,
        #[serde(default)]
        pub public_history: Vec<ChatMessage>,
        #[serde(default)]
        pub private_histories: HashMap<String, Vec<ChatMessage>>,
        #[serde(default)]
        pub generation_id: u64,
        #[serde(default)]
        pub created_at: i64,
    }

    impl Agent {
        pub fn new(name: &str, model: &str, prompt: &str, desc: &str) -> Self {
            Self {
                name: name.to_string(),
                description: desc.to_string(),
                model: model.to_string(),
                system_prompt: prompt.to_string(),
                public_history: Vec::new(),
                private_histories: HashMap::new(),
                generation_id: 0,
                created_at: chrono::Local::now().timestamp(),
            }
        }

        pub fn history_mut(&mut self, private: bool, uid: &str) -> &mut Vec<ChatMessage> {
            if private {
                self.private_histories.entry(uid.to_string()).or_default()
            } else {
                &mut self.public_history
            }
        }

        pub fn history(&self, private: bool, uid: &str) -> &[ChatMessage] {
            if private {
                self.private_histories
                    .get(uid)
                    .map(|v| v.as_slice())
                    .unwrap_or(&[])
            } else {
                &self.public_history
            }
        }

        pub fn clear_history(&mut self, private: bool, uid: &str) {
            if private {
                if let Some(h) = self.private_histories.get_mut(uid) {
                    h.clear();
                }
            } else {
                self.public_history.clear();
            }
        }

        pub fn delete_at(&mut self, private: bool, uid: &str, indices: &[usize]) -> Vec<usize> {
            let h = self.history_mut(private, uid);
            let mut deleted = Vec::new();
            let mut sorted: Vec<usize> = indices.to_vec();
            // 降序排序，从后往前删除
            sorted.sort_by(|a, b| b.cmp(a));
            sorted.dedup();
            for i in sorted {
                if i > 0 && i <= h.len() {
                    h.remove(i - 1);
                    deleted.push(i);
                }
            }
            // 返回时恢复升序，便于显示
            deleted.reverse();
            deleted
        }

        pub fn edit_at(&mut self, private: bool, uid: &str, idx: usize, content: &str) -> bool {
            let h = self.history_mut(private, uid);
            if idx > 0 && idx <= h.len() {
                h[idx - 1].content = content.to_string();
                true
            } else {
                false
            }
        }
    }

    #[derive(Debug, Clone, Serialize, Deserialize, Default)]
    pub struct Config {
        pub api_base: String,
        pub api_key: String,
        #[serde(default)]
        pub models: Vec<String>,
        #[serde(default)]
        pub agents: Vec<Agent>,
        #[serde(default)]
        pub default_model: String,
        #[serde(default)]
        pub default_prompt: String,
    }

    #[derive(Debug, Default)]
    pub struct GeneratingState {
        pub public: HashSet<String>,
        pub private: HashMap<String, HashSet<String>>,
    }

    impl GeneratingState {
        pub fn is_generating(&self, agent: &str, private: bool, uid: &str) -> bool {
            if private {
                self.private
                    .get(agent)
                    .map(|s| s.contains(uid))
                    .unwrap_or(false)
            } else {
                self.public.contains(agent)
            }
        }

        pub fn set_generating(&mut self, agent: &str, private: bool, uid: &str, generating: bool) {
            if private {
                let set = self.private.entry(agent.to_string()).or_default();
                if generating {
                    set.insert(uid.to_string());
                } else {
                    set.remove(uid);
                }
            } else if generating {
                self.public.insert(agent.to_string());
            } else {
                self.public.remove(agent);
            }
        }
    }
}

// --- 工具函数 ---
mod utils {
    use cdp_html_shot::{Browser, CaptureOptions, Viewport};
    use kovi::bot::message::Message;
    use kovi::tokio::time::{self, Duration};
    use pulldown_cmark::{Options, Parser, html};
    use regex::Regex;
    use std::sync::OnceLock;

    pub static RE_API: OnceLock<Regex> = OnceLock::new();
    pub static RE_IDX: OnceLock<Regex> = OnceLock::new();

    pub const MODEL_KEYWORDS: &[&str] =
        &["gpt-5", "claude", "gemini-3", "deepseek", "kimi", "grok-4"];

    /// 全角转半角
    pub fn normalize(s: &str) -> String {
        s.chars()
            .map(|c| match c {
                '！' => '!',
                '＠' => '@',
                '＃' => '#',
                '＄' => '$',
                '％' => '%',
                '＊' => '*',
                '（' => '(',
                '）' => ')',
                '－' => '-',
                '＋' => '+',
                '：' => ':',
                '；' => ';',
                '“' | '”' => '"',
                '‘' | '’' => '\'',
                '，' => ',',
                '。' => '.',
                '？' => '?',
                '～' => '~',
                '＿' => '_',
                '＆' => '&',
                '／' => '/',
                '＝' => '=',
                _ => c,
            })
            .collect()
    }

    /// 解析 API 配置
    pub fn parse_api(text: &str) -> Option<(String, String)> {
        let re = RE_API.get_or_init(|| {
            Regex::new(r"(?s)^(https?://\S+)\s+(sk-\S+)$|^(sk-\S+)\s+(https?://\S+)$").unwrap()
        });
        let t = text.trim();
        re.captures(t).and_then(|c| {
            c.get(1)
                .zip(c.get(2))
                .map(|(u, k)| (u.as_str().to_string(), k.as_str().to_string()))
                .or_else(|| {
                    c.get(3)
                        .zip(c.get(4))
                        .map(|(k, u)| (u.as_str().to_string(), k.as_str().to_string()))
                })
        })
    }

    /// 解析索引 (1, 1-5, 1,3,5)
    pub fn parse_indices(s: &str) -> Vec<usize> {
        let s = s.replace('，', ",");
        let re = RE_IDX.get_or_init(|| Regex::new(r"(\d+)(?:-(\d+))?").unwrap());
        let mut v = Vec::new();
        for c in re.captures_iter(&s) {
            if let Some(start) = c.get(1).and_then(|m| m.as_str().parse().ok()) {
                if let Some(end) = c.get(2).and_then(|m| m.as_str().parse().ok()) {
                    v.extend(start..=end);
                } else {
                    v.push(start);
                }
            }
        }
        v.sort();
        v.dedup();
        v
    }

    /// 过滤模型列表
    pub fn filter_models(models: &[String]) -> Vec<String> {
        models
            .iter()
            .filter(|m| {
                let lower = m.to_lowercase();
                MODEL_KEYWORDS.iter().any(|kw| lower.contains(kw))
            })
            .cloned()
            .collect()
    }

    pub fn escape_markdown_special(s: &str) -> String {
        // 使用 serde_json 转义特殊字符，然后去掉首尾引号
        match kovi::serde_json::to_string(s) {
            Ok(escaped) => {
                let trimmed = escaped.trim_matches('"');
                // 将 \n 还原为真实换行，保持可读性
                trimmed.replace("\\n", "\n").replace("\\t", "\t")
            }
            Err(_) => s.to_string(),
        }
    }

    pub async fn render_md(md: &str, title: &str) -> anyhow::Result<String> {
        let mut opts = Options::empty();
        opts.insert(Options::ENABLE_STRIKETHROUGH);
        opts.insert(Options::ENABLE_TABLES);
        let parser = Parser::new_ext(md, opts);
        let mut html_body = String::new();
        html::push_html(&mut html_body, parser);

        let css = r#"
 *{box-sizing:border-box}
 body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Hiragino Sans GB","Microsoft YaHei",Helvetica,Arial,sans-serif;font-size:15px;line-height:1.6;background:#f5f5f5;color:#333;padding:0;margin:0}
 .md{background:#fff;padding:16px 14px;margin:0;max-width:480px;width:90vw;word-wrap:break-word;overflow-wrap:break-word}
 .title{font-size:13px;color:#888;border-bottom:1px solid #eee;padding-bottom:10px;margin-bottom:14px;font-weight:500}
 h1,h2,h3{margin:16px 0 10px;font-weight:600;line-height:1.4}
 h1{font-size:20px;border-bottom:2px solid #eee;padding-bottom:8px}
 h2{font-size:18px;border-bottom:1px solid #eee;padding-bottom:6px}
 h3{font-size:16px}
 p{margin:10px 0}
 table{border-collapse:collapse;margin:12px 0;width:100%;font-size:13px;display:block;overflow-x:auto}
 td,th{padding:8px 10px;border:1px solid #ddd;text-align:left}
 th{font-weight:600;background:#f8f9fa}
 tr:nth-child(2n){background:#fafafa}
 code{padding:2px 6px;background:#f0f0f0;border-radius:4px;font-family:"SF Mono",Consolas,"Liberation Mono",Menlo,monospace;font-size:13px;color:#d63384;white-space:pre-wrap;word-wrap:break-word;}
 pre{background:#f6f8fa;border-radius:8px;padding:12px;overflow-x:auto;margin:12px 0;white-space:pre-wrap;word-wrap:break-word;overflow-wrap: break-word;}
 pre code{background:none;padding:0;color:#333}
 blockquote{margin:12px 0;padding:8px 12px;color:#666;border-left:3px solid #ddd;background:#fafafa;border-radius:0 4px 4px 0}
 img{max-width:100%;height:auto;border-radius:6px;margin:8px 0}
 ul,ol{padding-left:20px;margin:10px 0}
 li{margin:4px 0}
 hr{border:none;border-top:1px solid #eee;margin:16px 0}
 a{color:#0066cc;text-decoration:none}
 strong{font-weight:600}
 .agent-card{background:#fafbfc;border:1px solid #e8e8e8;border-radius:8px;padding:12px;margin:10px 0}
 .agent-name{font-size:16px;font-weight:600;color:#333;margin-bottom:8px}
 .agent-info{font-size:13px;color:#666;line-height:1.8}
 .agent-info code{font-size:12px}
 "#;
        let html = format!(
            r#"<!DOCTYPE html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>{css}</style></head><body><div class="md"><div class="title">{title}</div>{html_body}</div></body></html>"#
        );

        let browser = Browser::instance().await;
        let tab = browser.new_tab().await?;

        // 1. 设置初始视口
        // 宽度 600 以适应 .md max-width: 480px 的卡片设计
        // device_scale_factor: 2.0 提升截图清晰度
        let width = 600;
        tab.set_viewport(&Viewport::new(width, 100).with_device_scale_factor(2.0))
            .await?;

        tab.set_content(&html).await?;

        // 等待渲染
        time::sleep(Duration::from_millis(200)).await;

        // 2. 获取实际内容高度并调整视口
        // 修复长截图时底部出现大片空白的 Bug (Chromium Issue)
        let height_js = "document.body.scrollHeight";
        let body_height = tab.evaluate(height_js).await?.as_f64().unwrap_or(800.0) as u32;

        // 设置新的视口高度以容纳所有内容
        let viewport = Viewport::new(width, body_height + 100).with_device_scale_factor(2.0);
        tab.set_viewport(&viewport).await?;

        // 等待 Resize 生效
        time::sleep(Duration::from_millis(100)).await;

        // 3. 截图
        // 显式传入 viewport 确保 screenshot 方法使用了正确的尺寸
        let opts = CaptureOptions::new()
            .with_viewport(viewport)
            .with_quality(90);

        let b64 = tab
            .find_element(".md")
            .await?
            .screenshot_with_options(opts)
            .await?;

        let _ = tab.close().await;
        Ok(b64)
    }

    /// 获取消息完整内容(含引用)
    pub async fn get_full_content(
        event: &std::sync::Arc<kovi::MsgEvent>,
        bot: &std::sync::Arc<kovi::RuntimeBot>,
    ) -> (String, Vec<String>) {
        let mut text = String::new();
        let mut imgs = Vec::new();

        if let Some(reply) = event.message.iter().find(|s| s.type_ == "reply")
            && let Some(id) = reply.data.get("id").and_then(|v| v.as_str())
            && let Ok(id) = id.parse::<i32>()
            && let Ok(ret) = bot.get_msg(id).await
            && let Some(msg_data) = ret.data.get("message")
        {
            let reply_msg = Message::from_value(msg_data.clone()).unwrap_or_default();
            text.push_str("【引用】\n");
            for seg in reply_msg.iter() {
                match seg.type_.as_str() {
                    "text" => {
                        if let Some(t) = seg.data.get("text").and_then(|v| v.as_str()) {
                            text.push_str(t);
                        }
                    }
                    "image" => {
                        text.push_str("[图片]");
                        if let Some(u) = seg.data.get("url").and_then(|v| v.as_str()) {
                            imgs.push(u.to_string());
                        }
                    }
                    _ => {}
                }
            }
            text.push_str("\n【/引用】\n");
        }

        for seg in event.message.iter() {
            match seg.type_.as_str() {
                "text" => {
                    if let Some(t) = seg.data.get("text").and_then(|v| v.as_str()) {
                        text.push_str(t);
                    }
                }
                "image" => {
                    if let Some(u) = seg.data.get("url").and_then(|v| v.as_str()) {
                        imgs.push(u.to_string());
                    }
                }
                _ => {}
            }
        }
        (text.trim().to_string(), imgs)
    }

    /// 格式化历史记录
    pub fn format_history(hist: &[super::types::ChatMessage], offset: usize) -> String {
        hist.iter()
            .enumerate()
            .map(|(i, m)| {
                let emoji = match m.role.as_str() {
                    "user" => "👤",
                    "assistant" => "🤖",
                    "system" => "⚙️",
                    _ => "❓",
                };
                let time = chrono::DateTime::from_timestamp(m.timestamp, 0)
                    .map(|t| {
                        use chrono::TimeZone;
                        chrono::Local
                            .from_utc_datetime(&t.naive_utc())
                            .format("%m-%d %H:%M")
                            .to_string()
                    })
                    .unwrap_or_default();
                let img_note = if m.images.is_empty() {
                    String::new()
                } else {
                    format!(" [{}图]", m.images.len())
                };
                format!(
                    "**#{} {} {}{}**\n{}",
                    offset + i + 1,
                    emoji,
                    time,
                    img_note,
                    m.content
                )
            })
            .collect::<Vec<_>>()
            .join("\n\n---\n\n")
    }

    /// 截断字符串
    pub fn truncate_str(s: &str, max_chars: usize) -> String {
        let chars: Vec<char> = s.chars().collect();
        if chars.len() <= max_chars {
            s.to_string()
        } else {
            chars[..max_chars].iter().collect::<String>() + "..."
        }
    }

    pub fn format_export_txt(
        agent_name: &str,
        model: &str,
        scope: &str,
        hist: &[super::types::ChatMessage],
    ) -> String {
        let mut content = String::new();
        let separator = "─".repeat(40);
        let thin_sep = "┄".repeat(40);

        // 头部信息
        content.push_str(&format!("┏{}┓\n", "━".repeat(40)));
        content.push_str(&format!("┃  智能体: {:<32}┃\n", agent_name));
        content.push_str(&format!("┃  模  型: {:<32}┃\n", model));
        content.push_str(&format!("┃  类  型: {:<32}┃\n", scope));
        content.push_str(&format!(
            "┃  导  出: {:<32}┃\n",
            chrono::Local::now().format("%Y-%m-%d %H:%M:%S")
        ));
        content.push_str(&format!("┃  记录数: {:<32}┃\n", hist.len()));
        content.push_str(&format!("┗{}┛\n\n", "━".repeat(40)));

        // 历史记录
        for (i, m) in hist.iter().enumerate() {
            let time = chrono::DateTime::from_timestamp(m.timestamp, 0)
                .map(|t| {
                    use chrono::TimeZone;
                    chrono::Local
                        .from_utc_datetime(&t.naive_utc())
                        .format("%Y-%m-%d %H:%M:%S")
                        .to_string()
                })
                .unwrap_or_else(|| "未知时间".to_string());

            let role_name = match m.role.as_str() {
                "user" => "👤 用户",
                "assistant" => "🤖 助手",
                "system" => "⚙️ 系统",
                _ => &m.role,
            };

            content.push_str(&format!("【#{} {} | {}】\n", i + 1, role_name, time));
            content.push_str(&format!("{}\n", thin_sep));
            content.push_str(&m.content);
            content.push('\n');

            if !m.images.is_empty() {
                content.push_str(&format!("\n📷 附图 ({} 张):\n", m.images.len()));
                for (j, url) in m.images.iter().enumerate() {
                    content.push_str(&format!("   {}. {}\n", j + 1, url));
                }
            }

            content.push_str(&format!("\n{}\n\n", separator));
        }

        content
    }
}

// --- 指令解析器 ---
mod parser {
    use super::utils::normalize;

    #[derive(Debug, Clone, Copy, PartialEq)]
    pub enum Scope {
        Public,
        Private,
    }

    #[derive(Debug, Clone, PartialEq, Default)]
    pub enum Action {
        Chat,
        Regenerate,
        Stop,
        #[default]
        Create,
        Copy,
        SetDesc,
        Delete,
        List,
        SetModel,
        SetPrompt,
        ViewPrompt,
        ListModels,
        ViewAll(Scope),
        ViewAt(Scope),
        Export(Scope),
        EditAt(Scope),
        DeleteAt(Scope),
        ClearHistory(Scope),
        ClearAllPublic,
        ClearEverything,
        Help,
    }

    #[derive(Debug, Clone)]
    pub struct Command {
        pub agent: String,
        pub action: Action,
        pub args: String,
        pub indices: Vec<usize>,
        pub private_reply: bool,
        pub text_mode: bool,
    }

    impl Command {
        pub fn new(agent: &str, action: Action) -> Self {
            Self {
                agent: agent.to_string(),
                action,
                args: String::new(),
                indices: Vec::new(),
                private_reply: false,
                text_mode: false,
            }
        }
    }

    pub fn parse_global(raw: &str) -> Option<Command> {
        let norm = normalize(raw.trim());

        if norm == "oai" {
            return Some(Command::new("", Action::Help));
        }

        if norm == "/" {
            return Some(Command::new("", Action::List));
        }

        if norm == "/%" {
            return Some(Command::new("", Action::ListModels));
        }

        if norm == "-*" {
            return Some(Command::new("", Action::ClearAllPublic));
        }

        if norm == "-*!" {
            return Some(Command::new("", Action::ClearEverything));
        }

        None
    }

    pub fn parse_create(raw: &str) -> Option<(String, String, String, String)> {
        let norm = normalize(raw.trim());
        if !norm.starts_with("#") {
            return None;
        }

        let after = &raw.trim()[norm.find("#").unwrap() + "#".len()..];

        let name_end = after
            .find(|c: char| c.is_whitespace() || c == '(' || c == '（')
            .unwrap_or(after.len());
        let name = after[..name_end].trim().to_string();
        if name.is_empty() {
            return None;
        }

        let rest = &after[name_end..];

        let (desc, after_desc) = if rest.starts_with('(') || rest.starts_with('（') {
            if let Some(pos) = rest.find(')').or_else(|| rest.find('）')) {
                (rest[1..pos].to_string(), &rest[pos + 1..])
            } else {
                (String::new(), rest)
            }
        } else {
            (String::new(), rest)
        };

        let parts: Vec<&str> = after_desc.split_whitespace().collect();
        let model = parts.first().unwrap_or(&"").to_string();
        let prompt = if parts.len() > 1 {
            parts[1..].join(" ")
        } else {
            String::new()
        };

        Some((name, desc, model, prompt))
    }

    pub fn parse_delete_agent(raw: &str, agents: &[String]) -> Option<String> {
        let norm = normalize(raw.trim());
        if !norm.starts_with("-#") {
            return None;
        }
        let name = norm[2..].trim();
        if agents.iter().any(|a| a.eq_ignore_ascii_case(name)) {
            Some(name.to_string())
        } else {
            None
        }
    }

    pub fn parse_agent_cmd(raw: &str, agents: &[String]) -> Option<Command> {
        let raw = raw.trim();
        if raw.is_empty() {
            return None;
        }

        let norm = normalize(raw);
        let chars: Vec<char> = norm.chars().collect();

        let mut char_idx = 0;
        let mut private_reply = false;
        let mut text_mode = false;

        while char_idx < chars.len() {
            match chars[char_idx] {
                '&' => {
                    private_reply = true;
                    char_idx += 1;
                }
                '"' => {
                    text_mode = true;
                    char_idx += 1;
                }
                _ => break,
            }
        }

        let byte_idx: usize = chars.iter().take(char_idx).map(|c| c.len_utf8()).sum();
        let content = &norm[byte_idx..];

        let mut agent_name = String::new();
        let mut match_char_len = 0;
        let mut sorted = agents.to_vec();
        sorted.sort_by_key(|b| std::cmp::Reverse(b.chars().count()));

        for name in &sorted {
            let name_lower = name.to_lowercase();
            let content_lower = content.to_lowercase();
            if content_lower.starts_with(&name_lower) {
                agent_name = name.clone();
                match_char_len = name.chars().count();
                break;
            }
        }

        if agent_name.is_empty() {
            return None;
        }

        let match_byte_len: usize = content
            .chars()
            .take(match_char_len)
            .map(|c| c.len_utf8())
            .sum();
        let suffix = content[match_byte_len..].trim();

        let raw_suffix = {
            let prefix_bytes: usize = raw.chars().take(char_idx).map(|c| c.len_utf8()).sum();
            let agent_bytes: usize = raw[prefix_bytes..]
                .chars()
                .take(match_char_len)
                .map(|c| c.len_utf8())
                .sum();
            raw[prefix_bytes + agent_bytes..].trim()
        };

        let (action, args, indices) = parse_suffix(suffix, raw_suffix, private_reply);

        Some(Command {
            agent: agent_name,
            action,
            args,
            indices,
            private_reply,
            text_mode,
        })
    }

    fn parse_suffix(norm: &str, raw: &str, has_priv_prefix: bool) -> (Action, String, Vec<usize>) {
        let s = norm.trim();
        let r = raw.trim();

        if s.is_empty() {
            return (Action::Chat, r.to_string(), vec![]);
        }

        if s == "~" || (s.starts_with('~') && !s.starts_with("~#") && !s.starts_with("~$")) {
            let arg = if s.len() > 1 {
                r.get(1..).unwrap_or("").trim()
            } else {
                ""
            };
            return (Action::Regenerate, arg.to_string(), vec![]);
        }

        if s == "!" {
            return (Action::Stop, String::new(), vec![]);
        }

        if s.starts_with("~#") {
            let arg = r.get(2..).unwrap_or("").trim();
            return (Action::Copy, arg.to_string(), vec![]);
        }

        if (s.starts_with(':') || s.starts_with('：'))
            && !s.starts_with(":/")
            && !s.starts_with("：/")
        {
            let skip_len = if r.starts_with('：') {
                '：'.len_utf8() // 3 bytes for Chinese full-width colon
            } else {
                ':'.len_utf8() // 1 byte for ASCII colon
            };
            let arg = r.get(skip_len..).unwrap_or("").trim();
            return (Action::SetDesc, arg.to_string(), vec![]);
        }

        if s.starts_with('%') {
            let arg = r.get(1..).unwrap_or("").trim();
            return (Action::SetModel, arg.to_string(), vec![]);
        }

        if s.starts_with('$') && s != "/$" {
            let arg = r.get(1..).unwrap_or("").trim();
            return (Action::SetPrompt, arg.to_string(), vec![]);
        }

        if s == "/$" {
            return (Action::ViewPrompt, String::new(), vec![]);
        }

        let (has_local_priv, clean, clean_raw) = if let Some(stripped) = s.strip_prefix('&') {
            (true, stripped, r.strip_prefix('&').unwrap_or("").trim())
        } else {
            (false, s, r)
        };

        let scope = if has_priv_prefix || has_local_priv {
            Scope::Private
        } else {
            Scope::Public
        };

        if clean == "/*" {
            return (Action::ViewAll(scope), String::new(), vec![]);
        }

        if clean.starts_with('/') && clean.len() > 1 {
            let idx_part = &clean[1..];
            let indices = super::utils::parse_indices(idx_part);
            if !indices.is_empty() {
                return (Action::ViewAt(scope), String::new(), indices);
            }
        }

        if clean == "_*" {
            return (Action::Export(scope), String::new(), vec![]);
        }

        if clean.starts_with('\'') {
            let parts: Vec<&str> = clean_raw.get(1..).unwrap_or("").splitn(2, ' ').collect();
            if !parts.is_empty() {
                let indices = super::utils::parse_indices(parts[0]);
                let content = parts.get(1).unwrap_or(&"").to_string();
                return (Action::EditAt(scope), content, indices);
            }
        }

        if clean == "-*" {
            return (Action::ClearHistory(scope), String::new(), vec![]);
        }

        if clean.starts_with('-') && clean.len() > 1 {
            let idx_part = &clean[1..];
            let indices = super::utils::parse_indices(idx_part);
            if !indices.is_empty() {
                return (Action::DeleteAt(scope), String::new(), indices);
            }
        }

        (Action::Chat, r.to_string(), vec![])
    }
}

// --- 数据管理 ---
mod data {
    use super::types::{Config, GeneratingState};
    use async_openai::Client;
    use async_openai::config::OpenAIConfig;
    use kovi::tokio::sync::RwLock;
    use kovi::utils::{load_json_data, save_json_data};
    use std::path::PathBuf;

    pub struct Manager {
        pub config: RwLock<Config>,
        pub generating: RwLock<GeneratingState>,
        path: PathBuf,
    }

    impl Manager {
        pub fn new(dir: PathBuf) -> Self {
            let path = dir.join("config.json");
            let default = Config {
                default_model: "gpt-4o".to_string(),
                default_prompt: "You are a helpful assistant.".to_string(),
                ..Default::default()
            };
            let config = load_json_data(default.clone(), path.clone()).unwrap_or(default);
            Self {
                config: RwLock::new(config),
                generating: RwLock::new(GeneratingState::default()),
                path,
            }
        }

        pub fn save(&self, cfg: &Config) {
            let _ = save_json_data(cfg, &self.path);
        }

        pub async fn fetch_models(&self) -> anyhow::Result<Vec<String>> {
            let (base, key) = {
                let c = self.config.read().await;
                (c.api_base.clone(), c.api_key.clone())
            };

            if base.is_empty() {
                return Err(anyhow::anyhow!("API未配置"));
            }

            let config = OpenAIConfig::new().with_api_base(base).with_api_key(key);

            let client = Client::with_config(config);

            let response = client.models().list().await?;

            // 提取模型 ID 并排序
            let mut models: Vec<String> = response.data.into_iter().map(|m| m.id).collect();

            models.sort();

            let filtered = super::utils::filter_models(&models);
            let final_models = if filtered.is_empty() {
                models
            } else {
                filtered
            };

            {
                let mut c = self.config.write().await;
                c.models = final_models.clone();
                self.save(&c);
            }
            Ok(final_models)
        }

        pub fn resolve_model(&self, input: &str, models: &[String]) -> Option<String> {
            if input.is_empty() {
                return None;
            }
            if let Ok(i) = input.parse::<usize>()
                && i > 0
                && i <= models.len()
            {
                return Some(models[i - 1].clone());
            }
            let lower = input.to_lowercase();
            for m in models {
                if m.to_lowercase().contains(&lower) {
                    return Some(m.clone());
                }
            }
            Some(input.to_string())
        }

        pub async fn agent_names(&self) -> Vec<String> {
            self.config
                .read()
                .await
                .agents
                .iter()
                .map(|a| a.name.clone())
                .collect()
        }
    }
}

// --- 业务逻辑 ---
mod logic {
    use super::data::Manager;
    use super::parser::{Action, Command, Scope};
    use super::types::{Agent, ChatMessage};
    use super::utils::{
        escape_markdown_special, format_export_txt, format_history, render_md, truncate_str,
    };
    use async_openai::{
        Client,
        config::OpenAIConfig,
        types::{
            ChatCompletionRequestAssistantMessageArgs, ChatCompletionRequestMessage,
            ChatCompletionRequestMessageContentPartImageArgs,
            ChatCompletionRequestMessageContentPartTextArgs,
            ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestUserMessageArgs,
            CreateChatCompletionRequestArgs, ImageUrlArgs,
        },
    };
    use kovi::bot::message::Message;
    use kovi_plugin_expand_napcat::NapCatApi;
    use regex::Regex;
    use std::{fs::File, io::Write, sync::Arc};

    async fn reply(event: &Arc<kovi::MsgEvent>, text: &str, text_mode: bool, header: &str) {
        if text_mode {
            event.reply(text);
            return;
        }
        match render_md(text, header).await {
            Ok(b64) => event.reply(Message::new().add_image(&format!("base64://{}", b64))),
            Err(_) => event.reply(text),
        }
    }

    fn extract_image_urls(content: &str) -> Vec<String> {
        let re = Regex::new(
            r"!\[.*?\]\((https?://[^\s\)]+)\)|(?:https?://[^\s]+\.(?:png|jpg|jpeg|gif|webp|bmp))",
        )
        .unwrap();
        re.captures_iter(content)
            .filter_map(|cap| cap.get(1).or(cap.get(0)).map(|m| m.as_str().to_string()))
            .collect()
    }

    async fn chat(
        name: &str,
        prompt: &str,
        imgs: Vec<String>,
        regen: bool,
        cmd: &Command,
        event: &Arc<kovi::MsgEvent>,
        mgr: &Arc<Manager>,
        _bot: &Arc<kovi::RuntimeBot>,
    ) {
        struct ChatContext<'a> {
            name: &'a str,
            prompt: &'a str,
            imgs: Vec<String>,
            regen: bool,
            cmd: &'a Command,
            event: &'a Arc<kovi::MsgEvent>,
            mgr: &'a Arc<Manager>,
        }

        async fn inner(ctx: ChatContext<'_>) {
            let is_priv_ctx = ctx.cmd.private_reply;
            let uid = ctx.event.user_id.to_string();

            {
                let generating = ctx.mgr.generating.read().await;
                if generating.is_generating(ctx.name, is_priv_ctx, &uid) {
                    ctx.event.reply("⏳ 正在生成中，请等待或使用 智能体! 停止");
                    return;
                }
            }

            let (agent, api) = {
                let c = ctx.mgr.config.read().await;
                let a = c.agents.iter().find(|a| a.name == ctx.name).cloned();
                (a, (c.api_base.clone(), c.api_key.clone()))
            };

            let agent = match agent {
                Some(a) => a,
                None => {
                    ctx.event.reply(format!("❌ 智能体 {} 不存在", ctx.name));
                    return;
                }
            };

            if api.0.is_empty() || api.1.is_empty() {
                ctx.event.reply("❌ API 未配置");
                return;
            }

            let mut hist = agent.history(is_priv_ctx, &uid).to_vec();

            if ctx.regen {
                if hist.last().map(|m| m.role == "assistant").unwrap_or(false) {
                    hist.pop();
                }
                if !ctx.prompt.is_empty() {
                    if hist.last().map(|m| m.role == "user").unwrap_or(false) {
                        hist.pop();
                    }
                    hist.push(ChatMessage::new("user", ctx.prompt, ctx.imgs.clone()));
                }
            } else {
                if ctx.prompt.is_empty() && ctx.imgs.is_empty() {
                    ctx.event.reply("💬 请输入内容");
                    return;
                }
                hist.push(ChatMessage::new("user", ctx.prompt, ctx.imgs.clone()));
            }

            let gen_id = {
                let mut c = ctx.mgr.config.write().await;
                if let Some(a) = c.agents.iter_mut().find(|a| a.name == ctx.name) {
                    *a.history_mut(is_priv_ctx, &uid) = hist.clone();
                    a.generation_id += 1;
                    let id = a.generation_id;
                    ctx.mgr.save(&c);
                    id
                } else {
                    return;
                }
            };

            {
                let mut generating = ctx.mgr.generating.write().await;
                generating.set_generating(ctx.name, is_priv_ctx, &uid, true);
            }

            let client =
                Client::with_config(OpenAIConfig::new().with_api_base(api.0).with_api_key(api.1));

            let mut msgs: Vec<ChatCompletionRequestMessage> = vec![];

            if !agent.system_prompt.is_empty() {
                msgs.push(
                    ChatCompletionRequestSystemMessageArgs::default()
                        .content(agent.system_prompt.clone())
                        .build()
                        .unwrap()
                        .into(),
                );
            }

            for m in &hist {
                if m.role == "user" {
                    let mut parts = Vec::new();
                    if !m.content.is_empty() {
                        parts.push(
                            ChatCompletionRequestMessageContentPartTextArgs::default()
                                .text(m.content.clone())
                                .build()
                                .unwrap()
                                .into(),
                        );
                    }
                    for url in &m.images {
                        parts.push(
                            ChatCompletionRequestMessageContentPartImageArgs::default()
                                .image_url(ImageUrlArgs::default().url(url).build().unwrap())
                                .build()
                                .unwrap()
                                .into(),
                        );
                    }
                    msgs.push(
                        ChatCompletionRequestUserMessageArgs::default()
                            .content(parts)
                            .build()
                            .unwrap()
                            .into(),
                    );
                } else if m.role == "assistant" {
                    msgs.push(
                        ChatCompletionRequestAssistantMessageArgs::default()
                            .content(m.content.clone())
                            .build()
                            .unwrap()
                            .into(),
                    );
                }
            }

            let req = match CreateChatCompletionRequestArgs::default()
                .model(&agent.model)
                .messages(msgs)
                .build()
            {
                Ok(r) => r,
                Err(e) => {
                    let mut generating = ctx.mgr.generating.write().await;
                    generating.set_generating(ctx.name, is_priv_ctx, &uid, false);
                    ctx.event.reply(format!("❌ 请求构建失败: {}", e));
                    return;
                }
            };

            match client.chat().create(req).await {
                Ok(res) => {
                    {
                        let mut generating = ctx.mgr.generating.write().await;
                        generating.set_generating(ctx.name, is_priv_ctx, &uid, false);
                    }

                    {
                        let c = ctx.mgr.config.read().await;
                        if let Some(a) = c.agents.iter().find(|a| a.name == ctx.name)
                            && a.generation_id != gen_id
                        {
                            return;
                        }
                    }

                    if let Some(choice) = res.choices.first()
                        && let Some(content) = &choice.message.content
                    {
                        let msg_index = {
                            let c = ctx.mgr.config.read().await;
                            if let Some(a) = c.agents.iter().find(|a| a.name == ctx.name) {
                                a.history(is_priv_ctx, &uid).len() + 1
                            } else {
                                0
                            }
                        };

                        {
                            let mut c = ctx.mgr.config.write().await;
                            if let Some(a) = c.agents.iter_mut().find(|a| a.name == ctx.name) {
                                a.history_mut(is_priv_ctx, &uid).push(ChatMessage::new(
                                    "assistant",
                                    content,
                                    vec![],
                                ));
                            }
                            ctx.mgr.save(&c);
                        }

                        let image_urls = extract_image_urls(content);

                        let header = format!(
                            "{} #{}回复{}",
                            agent.name,
                            msg_index,
                            if ctx.cmd.private_reply {
                                " (私有)"
                            } else {
                                ""
                            }
                        );

                        let display_content = if !image_urls.is_empty() && !ctx.cmd.text_mode {
                            let urls_text = image_urls
                                .iter()
                                .map(|u| format!("- {}", u))
                                .collect::<Vec<_>>()
                                .join("\n");
                            format!("{}\n\n---\n**图片链接:**\n{}", content, urls_text)
                        } else {
                            content.clone()
                        };

                        if ctx.cmd.text_mode && !image_urls.is_empty() {
                            let re = Regex::new(r"!\[.*?\]\((https?://[^\s\)]+)\)").unwrap();
                            let text = re.replace_all(content, "$1").to_string();

                            reply(ctx.event, &text, true, &header).await;
                            for url in &image_urls {
                                ctx.event.reply(Message::new().add_image(url));
                            }
                        } else {
                            reply(ctx.event, &display_content, ctx.cmd.text_mode, &header).await;
                        }
                    }
                }
                Err(e) => {
                    {
                        let mut generating = ctx.mgr.generating.write().await;
                        generating.set_generating(ctx.name, is_priv_ctx, &uid, false);
                    }
                    ctx.event.reply(format!("❌ API错误: {}", e));
                }
            }
        }

        inner(ChatContext {
            name,
            prompt,
            imgs,
            regen,
            cmd,
            event,
            mgr,
        })
        .await;
    }

    pub async fn execute(
        cmd: Command,
        prompt: String,
        imgs: Vec<String>,
        event: &Arc<kovi::MsgEvent>,
        mgr: &Arc<Manager>,
        bot: &Arc<kovi::RuntimeBot>,
    ) {
        let name = &cmd.agent;
        let uid = event.user_id.to_string();

        match cmd.action {
            Action::Chat => {
                chat(name, &prompt, imgs, false, &cmd, event, mgr, bot).await;
            }

            Action::Regenerate => {
                chat(name, &cmd.args, imgs, true, &cmd, event, mgr, bot).await;
            }

            Action::Stop => {
                let is_priv_ctx = cmd.private_reply;
                {
                    let mut generating = mgr.generating.write().await;
                    generating.set_generating(name, is_priv_ctx, &uid, false);
                }
                let mut c = mgr.config.write().await;
                if let Some(a) = c.agents.iter_mut().find(|a| a.name == *name) {
                    a.generation_id += 1;
                    mgr.save(&c);
                    event.reply("🛑 已停止");
                } else {
                    event.reply(format!("❌ 智能体 {} 不存在", name));
                }
            }

            Action::Copy => {
                if cmd.args.is_empty() {
                    event.reply("❌ 请指定新名称: 智能体~#新名称");
                    return;
                }
                let mut c = mgr.config.write().await;
                if c.agents.iter().any(|a| a.name == cmd.args) {
                    event.reply(format!("❌ {} 已存在", cmd.args));
                    return;
                }
                if let Some(src) = c.agents.iter().find(|a| a.name == *name).cloned() {
                    let mut new_agent = Agent::new(
                        &cmd.args,
                        &src.model,
                        &src.system_prompt,
                        &format!("复制自 {}", name),
                    );
                    new_agent.description = src.description.clone();
                    c.agents.push(new_agent);
                    mgr.save(&c);
                    event.reply(format!("✨ 已复制 {} → {}", name, cmd.args));
                } else {
                    event.reply(format!("❌ {} 不存在", name));
                }
            }

            Action::SetDesc => {
                if cmd.args.is_empty() {
                    event.reply("❌ 请提供描述: 智能体:描述内容");
                    return;
                }
                let mut c = mgr.config.write().await;
                if let Some(a) = c.agents.iter_mut().find(|a| a.name == *name) {
                    a.description = cmd.args.clone();
                    mgr.save(&c);
                    event.reply(format!("📝 {} 描述已更新", name));
                } else {
                    event.reply(format!("❌ {} 不存在", name));
                }
            }

            Action::SetModel => {
                if cmd.args.is_empty() {
                    event.reply("❌ 请指定模型: 智能体%模型名");
                    return;
                }
                let mut c = mgr.config.write().await;
                let models = c.models.clone();
                if let Some(model) = mgr.resolve_model(&cmd.args, &models) {
                    if let Some(a) = c.agents.iter_mut().find(|a| a.name == *name) {
                        let old = a.model.clone();
                        a.model = model.clone();
                        mgr.save(&c);
                        event.reply(format!("🔄 {} 模型: {} → {}", name, old, model));
                    } else {
                        event.reply(format!("❌ {} 不存在", name));
                    }
                } else {
                    event.reply("❌ 无效模型");
                }
            }

            Action::SetPrompt => {
                let mut c = mgr.config.write().await;
                if let Some(a) = c.agents.iter_mut().find(|a| a.name == *name) {
                    a.system_prompt = cmd.args.clone();
                    mgr.save(&c);
                    if cmd.args.is_empty() {
                        event.reply(format!("📝 {} 提示词已清空", name));
                    } else {
                        event.reply(format!("📝 {} 提示词已更新", name));
                    }
                } else {
                    event.reply(format!("❌ {} 不存在", name));
                }
            }

            Action::ViewPrompt => {
                let c = mgr.config.read().await;
                if let Some(a) = c.agents.iter().find(|a| a.name == *name) {
                    let prompt_display = if a.system_prompt.is_empty() {
                        "(空)".to_string()
                    } else {
                        escape_markdown_special(&a.system_prompt)
                    };
                    let content = format!(
                        "**模型**: `{}`\n\n**提示词**:\n```\n{}\n```",
                        a.model, prompt_display
                    );
                    reply(
                        event,
                        &content,
                        cmd.text_mode,
                        &format!("{} 系统提示词", a.name),
                    )
                    .await;
                } else {
                    event.reply(format!("❌ {} 不存在", name));
                }
            }

            Action::List => {
                let c = mgr.config.read().await;
                if c.agents.is_empty() {
                    event.reply("📋 暂无智能体，使用 #名称 模型 提示词 创建");
                    return;
                }
                let mut sorted_agents = c.agents.clone();
                sorted_agents.sort_by(|a, b| a.name.to_lowercase().cmp(&b.name.to_lowercase()));
                let list = sorted_agents
                    .iter()
                    .enumerate()
                    .map(|(i, a)| {
                        let desc = if a.description.is_empty() {
                            "无描述".to_string()
                        } else {
                            truncate_str(&a.description, 30)
                        };
                        let prompt_preview = if a.system_prompt.is_empty() {
                            "无提示词".to_string()
                        } else {
                            truncate_str(&a.system_prompt, 40)
                        };
                        format!(
                            "<div class=\"agent-card\">\n<div class=\"agent-name\">{}. {}</div>\n<div class=\"agent-info\">\n📦 <code>{}</code><br>\n📝 {}<br>\n💬 {}\n</div>\n</div>",
                            i + 1,
                            a.name,
                            a.model,
                            desc,
                            prompt_preview
                        )
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                reply(
                    event,
                    &list,
                    cmd.text_mode,
                    &format!("📋 智能体列表 (共{}个)", c.agents.len()),
                )
                .await;
            }

            Action::Delete => {
                let mut c = mgr.config.write().await;
                if let Some(idx) = c.agents.iter().position(|a| a.name == *name) {
                    c.agents.remove(idx);
                    mgr.save(&c);
                    event.reply(format!("🗑️ 已删除 {}", name));
                } else {
                    event.reply(format!("❌ {} 不存在", name));
                }
            }

            Action::ListModels => {
                let c = mgr.config.read().await;
                if c.models.is_empty() {
                    drop(c);
                    event.reply("⏳ 正在获取模型列表...");
                    match mgr.fetch_models().await {
                        Ok(models) => {
                            let list = models
                                .iter()
                                .enumerate()
                                .map(|(i, m)| format!("{}. `{}`", i + 1, m))
                                .collect::<Vec<_>>()
                                .join("\n");
                            reply(
                                event,
                                &list,
                                cmd.text_mode,
                                &format!("📋 可用模型 (共{}个)", models.len()),
                            )
                            .await;
                        }
                        Err(e) => event.reply(format!("❌ 获取失败: {}", e)),
                    }
                } else {
                    let list = c
                        .models
                        .iter()
                        .enumerate()
                        .map(|(i, m)| format!("{}. `{}`", i + 1, m))
                        .collect::<Vec<_>>()
                        .join("\n");
                    reply(
                        event,
                        &list,
                        cmd.text_mode,
                        &format!("📋 模型列表 (共{}个)", c.models.len()),
                    )
                    .await;
                }
            }

            Action::ViewAll(scope) => {
                let c = mgr.config.read().await;
                if let Some(a) = c.agents.iter().find(|a| a.name == *name) {
                    let priv_scope = matches!(scope, Scope::Private);
                    let hist = a.history(priv_scope, &uid);
                    if hist.is_empty() {
                        let s = if priv_scope { "私有" } else { "公有" };
                        event.reply(format!("📭 {} {}历史为空", name, s));
                        return;
                    }
                    let content = format_history(hist, 0);
                    let header = format!(
                        "{} {}历史 ({} 条)",
                        name,
                        if priv_scope { "私有" } else { "公有" },
                        hist.len()
                    );
                    reply(event, &content, cmd.text_mode, &header).await;
                } else {
                    event.reply(format!("❌ {} 不存在", name));
                }
            }

            Action::ViewAt(scope) => {
                if cmd.indices.is_empty() {
                    event.reply("❌ 请指定索引: 智能体/索引");
                    return;
                }
                let c = mgr.config.read().await;
                if let Some(a) = c.agents.iter().find(|a| a.name == *name) {
                    let priv_scope = matches!(scope, Scope::Private);
                    let hist = a.history(priv_scope, &uid);
                    let mut results = Vec::new();
                    for i in &cmd.indices {
                        if *i > 0 && *i <= hist.len() {
                            let m = &hist[i - 1];
                            let emoji = match m.role.as_str() {
                                "user" => "👤",
                                "assistant" => "🤖",
                                _ => "❓",
                            };
                            results.push(format!("**#{} {}**\n{}", i, emoji, m.content));
                        }
                    }
                    if results.is_empty() {
                        event.reply("❌ 索引无效");
                    } else {
                        reply(
                            event,
                            &results.join("\n\n---\n\n"),
                            cmd.text_mode,
                            &format!("{} 历史记录", name),
                        )
                        .await;
                    }
                } else {
                    event.reply(format!("❌ {} 不存在", name));
                }
            }

            Action::Export(scope) => {
                let c = mgr.config.read().await;
                if let Some(a) = c.agents.iter().find(|a| a.name == *name) {
                    let priv_scope = matches!(scope, Scope::Private);
                    let hist = a.history(priv_scope, &uid);
                    if hist.is_empty() {
                        event.reply("📭 历史为空");
                        return;
                    }

                    let scope_str = if priv_scope { "私有" } else { "公有" };
                    let content = format_export_txt(name, &a.model, scope_str, hist);

                    let scope_file = if priv_scope { "private" } else { "public" };
                    let fname = format!(
                        "{}_{}_{}_{}.txt",
                        name,
                        scope_file,
                        uid,
                        chrono::Local::now().format("%Y%m%d%H%M%S")
                    );
                    let path = bot.get_data_path().join(&fname);
                    match File::create(&path) {
                        Ok(mut f) => {
                            if f.write_all(content.as_bytes()).is_ok() {
                                let path_str = path.to_string_lossy().to_string();
                                let result = if let Some(gid) = event.group_id {
                                    bot.upload_group_file(gid, &path_str, &fname, None).await
                                } else {
                                    bot.upload_private_file(event.user_id, &path_str, &fname)
                                        .await
                                };
                                match result {
                                    Ok(_) => event.reply(format!("📤 已导出: {}", fname)),
                                    Err(e) => event.reply(format!("❌ 上传失败: {}", e)),
                                }
                            } else {
                                event.reply("❌ 写入失败");
                            }
                        }
                        Err(e) => event.reply(format!("❌ 创建文件失败: {}", e)),
                    }
                } else {
                    event.reply(format!("❌ {} 不存在", name));
                }
            }

            Action::EditAt(scope) => {
                if cmd.indices.is_empty() {
                    event.reply("❌ 请指定索引: 智能体'索引 新内容");
                    return;
                }
                if cmd.args.is_empty() {
                    event.reply("❌ 请提供新内容");
                    return;
                }
                let idx = cmd.indices[0];
                let mut c = mgr.config.write().await;
                if let Some(a) = c.agents.iter_mut().find(|a| a.name == *name) {
                    let priv_scope = matches!(scope, Scope::Private);
                    if a.edit_at(priv_scope, &uid, idx, &cmd.args) {
                        mgr.save(&c);
                        event.reply(format!("✏️ 已编辑第 {} 条", idx));
                    } else {
                        event.reply(format!("❌ 索引 {} 无效", idx));
                    }
                } else {
                    event.reply(format!("❌ {} 不存在", name));
                }
            }

            Action::DeleteAt(scope) => {
                if cmd.indices.is_empty() {
                    event.reply("❌ 请指定索引: 智能体-索引 (支持 1,3,5 或 1-5)");
                    return;
                }
                let mut c = mgr.config.write().await;
                if let Some(a) = c.agents.iter_mut().find(|a| a.name == *name) {
                    let priv_scope = matches!(scope, Scope::Private);
                    let deleted = a.delete_at(priv_scope, &uid, &cmd.indices);
                    if deleted.is_empty() {
                        event.reply("❌ 索引无效");
                    } else {
                        mgr.save(&c);
                        let s = deleted
                            .iter()
                            .map(|i| i.to_string())
                            .collect::<Vec<_>>()
                            .join(", ");
                        event.reply(format!("🗑️ 已删除第 {} 条 (共{}条)", s, deleted.len()));
                    }
                } else {
                    event.reply(format!("❌ {} 不存在", name));
                }
            }

            Action::ClearHistory(scope) => {
                let is_priv_ctx = cmd.private_reply;
                {
                    let mut generating = mgr.generating.write().await;
                    generating.set_generating(name, is_priv_ctx, &uid, false);
                }
                let mut c = mgr.config.write().await;
                if let Some(a) = c.agents.iter_mut().find(|a| a.name == *name) {
                    let priv_scope = matches!(scope, Scope::Private);
                    let s = if priv_scope { "私有" } else { "公有" };
                    a.clear_history(priv_scope, &uid);
                    a.generation_id += 1;
                    mgr.save(&c);
                    event.reply(format!("🔥 {} {}历史已清空", name, s));
                } else {
                    event.reply(format!("❌ {} 不存在", name));
                }
            }

            Action::ClearAllPublic => {
                {
                    let mut generating = mgr.generating.write().await;
                    generating.public.clear();
                }
                let mut c = mgr.config.write().await;
                let cnt = c.agents.len();
                for a in c.agents.iter_mut() {
                    a.public_history.clear();
                    a.generation_id += 1;
                }
                mgr.save(&c);
                event.reply(format!("🧹 已清空 {} 个智能体的公有历史", cnt));
            }

            Action::ClearEverything => {
                {
                    let mut generating = mgr.generating.write().await;
                    generating.public.clear();
                    generating.private.clear();
                }
                let mut c = mgr.config.write().await;
                let cnt = c.agents.len();
                for a in c.agents.iter_mut() {
                    a.public_history.clear();
                    a.private_histories.clear();
                    a.generation_id += 1;
                }
                mgr.save(&c);
                event.reply(format!("☢️ 已清空 {} 个智能体的所有历史", cnt));
            }

            Action::Help => {
                let help = r#"## 模式前缀（可组合）
| 符号 | 含义 |
|:---:|------|
| `&` | 私有模式 |
| `"` | 文本模式 |

## 智能体管理
| 指令 | 功能 | 示例 |
|------|------|------|
| `#名称 模型 提示词` | 创建/更新 | `#助手 gpt-4o 你是助手` |
| `智能体~#新名` | 复制 | `助手~#助手2` |
| `智能体:描述` | 设置描述 | `助手:通用助手` |
| `-#名称` | 删除 | `-#助手` |
| `/` | 列表 | `/` |

## 配置修改
| 指令 | 功能 | 示例 |
|------|------|------|
| `智能体%模型` | 修改模型 | `助手%gpt-4` |
| `智能体$提示词` | 修改提示词 | `助手$你是...` |
| `智能体$` | 清空提示词 | `助手$` |
| `智能体/$` | 查看提示词 | `助手/$` |
| `/%` | 模型列表 | `/%` |

## 对话控制
| 指令 | 功能 |
|------|------|
| `智能体 内容` | 对话 |
| `"智能体 内容` | 文本模式对话 |
| `&智能体 内容` | 私有对话 |
| `智能体~` | 重新生成 |
| `智能体!` | 停止生成 |

## 历史管理
| 指令 | 功能 |
|------|------|
| `智能体/*` | 查看所有 |
| `智能体/1` | 查看第1条 |
| `智能体/1-5` | 查看1-5条 |
| `智能体_*` | 导出(.txt) |
| `智能体'1 新内容` | 编辑第1条 |
| `智能体-1` | 删除第1条 |
| `智能体-1,3,5` | 删除多条 |
| `智能体-1-5` | 删除范围 |
| `智能体-*` | 清空历史 |

> 加 `&` 前缀操作私有历史: `&智能体/*`

## 危险操作
| 指令 | 功能 |
|------|------|
| `-*` | 清空所有智能体公有历史 |
| `-*!` | 清空所有历史 |

## API 配置
直接发送: `API地址 API密钥`
"#;
                reply(event, help, cmd.text_mode, "🤖 OAI 符号指令帮助").await;
            }

            Action::Create => {}
        }
    }

    pub async fn handle_create(
        name: &str,
        desc: &str,
        model: &str,
        prompt: &str,
        event: &Arc<kovi::MsgEvent>,
        mgr: &Arc<Manager>,
    ) {
        let mut c = mgr.config.write().await;
        let models = c.models.clone();

        let model = mgr
            .resolve_model(model, &models)
            .unwrap_or_else(|| model.to_string());

        let prompt = if prompt.is_empty() && !c.agents.iter().any(|a| a.name == name) {
            c.default_prompt.clone()
        } else {
            prompt.to_string()
        };

        if let Some(a) = c.agents.iter_mut().find(|a| a.name == name) {
            if !model.is_empty() {
                a.model = model.clone();
            }
            a.system_prompt = prompt;
            if !desc.is_empty() {
                a.description = desc.to_string();
            }
            let updated_model = a.model.clone();
            mgr.save(&c);
            event.reply(format!("📝 已更新 {} (模型: {})", name, updated_model));
        } else {
            let description = if desc.is_empty() {
                "新建智能体".to_string()
            } else {
                desc.to_string()
            };
            c.agents
                .push(Agent::new(name, &model, &prompt, &description));
            mgr.save(&c);
            event.reply(format!("✨ 已创建 {} (模型: {})", name, model));
        }
    }
}

// --- 入口 ---
use kovi::PluginBuilder;
use std::sync::Arc;

#[kovi::plugin]
async fn main() {
    let bot = PluginBuilder::get_runtime_bot();
    let mgr = Arc::new(data::Manager::new(bot.get_data_path()));

    let m = mgr.clone();
    kovi::tokio::spawn(async move {
        let _ = m.fetch_models().await;
    });

    let mgr_clone = mgr.clone();
    PluginBuilder::on_msg(move |event| {
        let mgr = mgr_clone.clone();
        let bot = bot.clone();
        async move {
            let raw = match event.borrow_text() {
                Some(v) => v,
                None => return,
            };

            if let Some((url, key)) = utils::parse_api(raw) {
                let mut c = mgr.config.write().await;
                c.api_base = url.clone();
                c.api_key = key;
                mgr.save(&c);
                drop(c);
                event.reply(format!("✅ API 已配置: {}", url));
                match mgr.fetch_models().await {
                    Ok(models) => event.reply(format!("📋 已获取 {} 个模型", models.len())),
                    Err(e) => event.reply(format!("⚠️ 获取模型失败: {}", e)),
                }
                return;
            }

            if let Some(cmd) = parser::parse_global(raw) {
                logic::execute(cmd, String::new(), vec![], &event, &mgr, &bot).await;
                return;
            }

            if let Some((name, desc, model, prompt)) = parser::parse_create(raw) {
                logic::handle_create(&name, &desc, &model, &prompt, &event, &mgr).await;
                return;
            }

            let agents = mgr.agent_names().await;
            if let Some(name) = parser::parse_delete_agent(raw, &agents) {
                let cmd = parser::Command::new(&name, parser::Action::Delete);
                logic::execute(cmd, String::new(), vec![], &event, &mgr, &bot).await;
                return;
            }

            if let Some(cmd) = parser::parse_agent_cmd(raw, &agents) {
                let (full_text, imgs) = utils::get_full_content(&event, &bot).await;
                let prompt = if matches!(
                    cmd.action,
                    parser::Action::Chat | parser::Action::Regenerate
                ) {
                    if full_text.contains("【引用") {
                        if cmd.args.is_empty() {
                            full_text
                        } else {
                            format!("{}\n{}", full_text, cmd.args)
                        }
                    } else {
                        cmd.args.clone()
                    }
                } else {
                    cmd.args.clone()
                };
                logic::execute(cmd, prompt, imgs, &event, &mgr, &bot).await;
            }
        }
    });
}
