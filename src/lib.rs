//! kovi-plugin-oai
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

    pub const MODEL_KEYWORDS: &[&str] = &[
        "gpt-5", "claude", "gemini-3", "deepseek", "kimi", "grok-4", "banana", "sora-2",
    ];

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
 .model-group{margin-bottom:16px;break-inside:avoid;}
 .model-header{background:#f0f2f5;color:#444;padding:6px 10px;border-radius:6px;font-weight:600;font-size:13px;margin-bottom:8px;display:flex;justify-content:space-between;align-items:center;border-left:3px solid #0066cc;}
 .model-count{background:rgba(0,0,0,0.05);color:#666;font-size:11px;padding:1px 6px;border-radius:4px;}
 .agent-grid{display:grid;/*手机端一行两列，充分利用宽度*/grid-template-columns:repeat(2,1fr);gap:8px;}
 .agent-mini{background:#fff;border:1px solid #eee;border-radius:6px;padding:8px;display:flex;flex-direction:column;justify-content:center;transition:background 0.2s;}
 .agent-mini-top{display:flex;align-items:center;margin-bottom:4px;}
 .agent-idx{background:#e6f0ff;color:#0066cc;font-size:10px;font-weight:700;min-width:18px;height:18px;border-radius:4px;display:flex;align-items:center;justify-content:center;margin-right:6px;flex-shrink:0;}
 .agent-mini-name{font-size:14px;font-weight:600;color:#333;overflow:hidden;white-space:nowrap;text-overflow:ellipsis;}
 .agent-mini-desc{font-size:11px;color:#999;overflow:hidden;white-space:nowrap;text-overflow:ellipsis;}
 .provider-section { margin-bottom: 20px; break-inside: avoid; }
 .provider-title { font-size: 14px; font-weight: 700; color: #555; margin-bottom: 8px; padding-left: 4px; border-left: 3px solid #666; line-height: 1.2; }
 .chip-container { display: flex; flex-wrap: wrap; gap: 8px; }
 .chip { background: #fff; border: 1px solid #ddd; border-radius: 6px; padding: 6px 10px; display: flex; align-items: center; font-size: 13px; color: #333; box-shadow: 0 1px 2px rgba(0,0,0,0.02); }
 .chip-idx { background: #f0f0f0; color: #666; font-size: 11px; padding: 2px 5px; border-radius: 4px; margin-right: 6px; font-family: monospace; font-weight: 600; }
 .chip-name { font-weight: 500; }
 .chip-badge { margin-left: 6px; background: #e6f0ff; color: #0066cc; font-size: 10px; padding: 1px 5px; border-radius: 10px; font-weight: 600; }

  .mod-group { margin-bottom: 16px; break-inside: avoid; }
  .mod-title { font-size: 13px; font-weight: 700; color: #666; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.5px; border-left: 3px solid #0066cc; padding-left: 6px; }
  .chip-box { display: flex; flex-wrap: wrap; gap: 8px; }
  .chip { background: #fff; border: 1px solid #e0e0e0; border-radius: 6px; padding: 6px 10px; display: flex; align-items: center; font-size: 13px; color: #333; transition: all 0.2s; }
  .chip-idx { background: #f5f5f5; color: #888; font-size: 11px; padding: 2px 6px; border-radius: 4px; margin-right: 8px; font-family: monospace; font-weight: 600; }
  .chip-name { font-weight: 500; }
  /* 正在使用的模型的徽标样式 */
  .chip-bad { margin-left: 8px; background: #e6f7ff; color: #1890ff; font-size: 10px; padding: 2px 6px; border-radius: 10px; font-weight: 600; } "#;
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
    /// 获取引用内容(格式化为 Markdown)及所有相关图片
    pub async fn get_full_content(
        event: &std::sync::Arc<kovi::MsgEvent>,
        bot: &std::sync::Arc<kovi::RuntimeBot>,
    ) -> (String, Vec<String>) {
        let mut quote_text = String::new();
        let mut imgs = Vec::new();

        // 1. 处理引用消息 (Reply)
        if let Some(reply) = event.message.iter().find(|s| s.type_ == "reply")
            && let Some(id) = reply.data.get("id").and_then(|v| v.as_str())
            && let Ok(id) = id.parse::<i32>()
            && let Ok(ret) = bot.get_msg(id).await
            && let Some(msg_data) = ret.data.get("message")
        {
            let reply_msg = Message::from_value(msg_data.clone()).unwrap_or_default();
            let mut temp_text = String::new();

            for seg in reply_msg.iter() {
                match seg.type_.as_str() {
                    "text" => {
                        if let Some(t) = seg.data.get("text").and_then(|v| v.as_str()) {
                            temp_text.push_str(t);
                        }
                    }
                    "image" => {
                        // 引用图片仅添加到图片列表，不再在文本中插入 "[图片]" 标记
                        if let Some(u) = seg.data.get("url").and_then(|v| v.as_str()) {
                            imgs.push(u.to_string());
                        }
                    }
                    "video" => {
                        // 尝试获取 url 或 file 字段
                        let url = seg
                            .data
                            .get("url")
                            .or(seg.data.get("file"))
                            .and_then(|v| v.as_str());
                        if let Some(u) = url {
                            imgs.push(u.to_string());
                        }
                    }
                    _ => {}
                }
            }

            // 使用 Markdown 引用块 "> "
            // 且如果 temp_text 为空（纯图片引用），则不添加任何引用文本
            let trimmed = temp_text.trim();
            if !trimmed.is_empty() {
                for line in trimmed.lines() {
                    quote_text.push_str("> ");
                    quote_text.push_str(line);
                    quote_text.push('\n');
                }
                quote_text.push('\n'); // 引用块与正文的分隔
            }
        }

        // 2. 提取当前消息中的图片/视频
        for seg in event.message.iter() {
            if seg.type_ == "image"
                && let Some(u) = seg.data.get("url").and_then(|v| v.as_str())
            {
                imgs.push(u.to_string());
            } else if seg.type_ == "video" {
                let url = seg
                    .data
                    .get("url")
                    .or(seg.data.get("file"))
                    .and_then(|v| v.as_str());
                if let Some(u) = url {
                    imgs.push(u.to_string());
                }
            }
        }

        // 返回 (引用文本, 所有图片URL)
        (quote_text, imgs)
    }

    /// 格式化历史记录
    pub fn format_history(
        hist: &[super::types::ChatMessage],
        offset: usize,
        text_mode: bool,
    ) -> String {
        let re = Regex::new(r"!\[.*?\]\((data:image/[^\s\)]+)\)").unwrap();

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
                    .map(|dt| {
                        use chrono::TimeZone;
                        chrono::Local
                            .from_utc_datetime(&dt.naive_utc())
                            .format("%m-%d %H:%M")
                            .to_string()
                    })
                    .unwrap_or_default();

                let mut body = m.content.clone();

                if text_mode {
                    body = re.replace_all(&body, "[图片]").to_string();
                }

                if !m.images.is_empty() {
                    if !body.is_empty() {
                        body.push_str("\n\n");
                    }

                    if text_mode {
                        let links = m
                            .images
                            .iter()
                            .map(|u| {
                                if u.starts_with("data:") {
                                    "- [Base64 Image]".to_string()
                                } else {
                                    format!("- [图片] {}", u)
                                }
                            })
                            .collect::<Vec<_>>()
                            .join("\n");
                        body.push_str(&links);
                    } else {
                        let imgs = m
                            .images
                            .iter()
                            .map(|u| format!("![image]({})", u))
                            .collect::<Vec<_>>()
                            .join("\n");
                        body.push_str(&imgs);
                    }
                }

                if body.trim().is_empty() {
                    body = "(无内容)".to_string();
                }

                format!("**#{} {} {}**\n{}", offset + i + 1, emoji, time, body)
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
        let re = Regex::new(r"!\[.*?\]\((data:image/[^\s\)]+)\)").unwrap();

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

            let clean_content = re.replace_all(&m.content, "[图片数据]");
            content.push_str(&clean_content);
            content.push('\n');

            if !m.images.is_empty() {
                content.push_str(&format!("\n📷 附图 ({} 张):\n", m.images.len()));
                for (j, url) in m.images.iter().enumerate() {
                    if url.starts_with("data:") {
                        content.push_str(&format!("   {}. [Base64 Image Data]\n", j + 1));
                    } else {
                        content.push_str(&format!("   {}. {}\n", j + 1, url));
                    }
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
        Rename,
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
        AutoFillDescriptions(String),
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

        if norm == "/#" {
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

        if norm.starts_with("##:") {
            let args = norm.get(3..).unwrap_or("").trim().to_string();
            return Some(Command::new("", Action::AutoFillDescriptions(args)));
        }

        None
    }

    pub fn parse_create(raw: &str) -> Option<(String, String, String, String)> {
        let norm = normalize(raw.trim());
        if !norm.starts_with("##") {
            return None;
        }

        let start_pos = norm.find("##").unwrap() + "##".len();
        let after = &raw.trim()[start_pos..];

        let name_end = after
            .find(|c: char| c.is_whitespace() || c == '(' || c == '（')
            .unwrap_or(after.len());
        let name = after[..name_end].trim().to_string();

        if name.is_empty()
            || name.chars().count() > 7
            || name.chars().any(|c| "&\"#~/ -_'!@$%:*".contains(c))
        {
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
        if model.chars().count() > 50 {
            return None;
        }
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

        if (s == "~" || s == "～")
            || ((s.starts_with('~') || s.starts_with('～'))
                && !s.starts_with("~#")
                && !s.starts_with("~$")
                && !s.starts_with("～#")
                && !s.starts_with("～$"))
        {
            let skip_len = if s.starts_with('～') {
                '～'.len_utf8()
            } else {
                '~'.len_utf8()
            };
            let arg = r.get(skip_len..).unwrap_or("").trim();
            return (Action::Regenerate, arg.to_string(), vec![]);
        }

        if s == "!" {
            return (Action::Stop, String::new(), vec![]);
        }

        if s.starts_with("~#") || s.starts_with("~＃") {
            let skip_len = if r.starts_with("~＃") {
                "～＃".chars().map(|c| c.len_utf8()).sum()
            } else {
                "~#".chars().map(|c| c.len_utf8()).sum()
            };
            let arg = r.get(skip_len..).unwrap_or("").trim();
            return (Action::Copy, arg.to_string(), vec![]);
        }

        if s.starts_with("~=") || s.starts_with("~＝") {
            let skip_len = if r.starts_with("~＝") {
                "~＝".chars().map(|c| c.len_utf8()).sum()
            } else {
                "~=".chars().map(|c| c.len_utf8()).sum()
            };
            let arg = r.get(skip_len..).unwrap_or("").trim();
            return (Action::Rename, arg.to_string(), vec![]);
        }

        if (s.starts_with(':') || s.starts_with('：'))
            && !s.starts_with(":/")
            && !s.starts_with("：/")
        {
            let skip_len = if r.starts_with('：') {
                '：'.len_utf8()
            } else {
                ':'.len_utf8()
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
    use crate::utils::truncate_str;

    use super::data::Manager;
    use super::parser::{Action, Command, Scope};
    use super::types::{Agent, ChatMessage};
    use super::utils::{escape_markdown_special, format_export_txt, format_history, render_md};
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

    pub(crate) fn reply_text(event: &Arc<kovi::MsgEvent>, text: impl Into<String>) {
        event.reply(
            Message::new()
                .add_reply(event.message_id)
                .add_text(text.into()),
        );
    }

    async fn reply(event: &Arc<kovi::MsgEvent>, text: &str, text_mode: bool, header: &str) {
        let msg = Message::new().add_reply(event.message_id);

        if text_mode {
            event.reply(msg.add_text(text));
            return;
        }
        match render_md(text, header).await {
            Ok(b64) => event.reply(msg.add_image(&format!("base64://{}", b64))),
            Err(_) => {
                let re = Regex::new(r"!\[.*?\]\((data:image/[^\s\)]+)\)").unwrap();
                let clean_text = re.replace_all(text, "[图片渲染失败]").to_string();
                event.reply(msg.add_text(&clean_text));
            }
        }
    }

    fn extract_image_urls(content: &str) -> Vec<String> {
        let re = Regex::new(
                    r"!\[.*?\]\(((?:https?://|data:image/)[^\s\)]+)\)|(?:https?://[^\s]+\.(?:png|jpg|jpeg|gif|webp|bmp))",
                )
                .unwrap();

        let mut urls: Vec<String> = re
            .captures_iter(content)
            .filter_map(|cap| cap.get(1).or(cap.get(0)).map(|m| m.as_str().to_string()))
            .collect();

        let mut seen = std::collections::HashSet::new();
        urls.retain(|url| seen.insert(url.clone()));

        urls
    }

    fn extract_video_urls(content: &str) -> Vec<String> {
        // 匹配 [download video](url)
        let re = Regex::new(r"\[download video\]\((https?://[^\s\)]+)\)").unwrap();
        re.captures_iter(content)
            .filter_map(|cap| cap.get(1).map(|m| m.as_str().to_string()))
            .collect()
    }

    #[allow(clippy::too_many_arguments)]
    async fn chat(
        name: &str,
        prompt: &str,
        imgs: Vec<String>,
        regen: bool,
        cmd: &Command,
        event: &Arc<kovi::MsgEvent>,
        mgr: &Arc<Manager>,
        bot: &Arc<kovi::RuntimeBot>,
    ) {
        struct ChatContext<'a> {
            name: &'a str,
            prompt: &'a str,
            imgs: Vec<String>,
            regen: bool,
            cmd: &'a Command,
            event: &'a Arc<kovi::MsgEvent>,
            mgr: &'a Arc<Manager>,
            bot: &'a Arc<kovi::RuntimeBot>,
        }

        async fn inner(ctx: ChatContext<'_>) {
            let is_priv_ctx = ctx.cmd.private_reply;
            let uid = ctx.event.user_id.to_string();

            {
                let generating = ctx.mgr.generating.read().await;
                if generating.is_generating(ctx.name, is_priv_ctx, &uid) {
                    reply_text(ctx.event, "⏳ 正在生成中，请等待或使用 智能体! 停止");
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
                    reply_text(ctx.event, format!("❌ 智能体 {} 不存在", ctx.name));
                    return;
                }
            };

            if api.0.is_empty() || api.1.is_empty() {
                reply_text(ctx.event, "❌ API 未配置");
                return;
            }

            match ctx
                .bot
                .set_msg_emoji_like(ctx.event.message_id.into(), "124")
                .await
            {
                Ok(_) => {
                    // kovi::log::info!("点赞成功");
                }
                Err(e) => {
                    kovi::log::error!("点赞失败: {:?}", e);
                }
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
                    reply_text(ctx.event, "💬 请输入内容");
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
            let re = Regex::new(r"!\[.*?\]\((data:image/[^\s\)]+)\)").unwrap();
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
                    if parts.is_empty() {
                        continue;
                    }
                    msgs.push(
                        ChatCompletionRequestUserMessageArgs::default()
                            .content(parts)
                            .build()
                            .unwrap()
                            .into(),
                    );
                } else if m.role == "assistant" {
                    let clean_content = re.replace_all(&m.content, "[Image Created]").to_string();

                    msgs.push(
                        ChatCompletionRequestAssistantMessageArgs::default()
                            .content(clean_content)
                            .build()
                            .unwrap()
                            .into(),
                    );

                    let gen_imgs = extract_image_urls(&m.content);
                    if !gen_imgs.is_empty() {
                        let mut img_parts = Vec::new();
                        for url in gen_imgs {
                            img_parts.push(
                                ChatCompletionRequestMessageContentPartImageArgs::default()
                                    .image_url(ImageUrlArgs::default().url(url).build().unwrap())
                                    .build()
                                    .unwrap()
                                    .into(),
                            );
                        }
                        msgs.push(
                            ChatCompletionRequestUserMessageArgs::default()
                                .content(img_parts)
                                .build()
                                .unwrap()
                                .into(),
                        );
                    }
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
                    reply_text(ctx.event, format!("❌ 请求构建失败: {}", e));
                    return;
                }
            };

            match kovi::tokio::time::timeout(
                std::time::Duration::from_secs(300),
                client.chat().create(req),
            )
            .await
            {
                // 情况 1: 触发超时 (超过 5 分钟)
                Err(_) => {
                    {
                        let mut generating = ctx.mgr.generating.write().await;
                        generating.set_generating(ctx.name, is_priv_ctx, &uid, false);
                    }
                    reply_text(
                        ctx.event,
                        "⏳ 请求超时：模型响应时间超过 5 分钟，已强制停止。",
                    );
                }
                // 情况 2: 请求在限时内完成 (包含 成功响应 或 API报错)
                Ok(result) => match result {
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
                                    .map(|u| {
                                        if u.starts_with("data:") {
                                            "- [Base64 Image]".to_string()
                                        } else {
                                            format!("- {}", u)
                                        }
                                    })
                                    .collect::<Vec<_>>()
                                    .join("\n");
                                format!("{}\n\n---\n**图片链接:**\n{}", content, urls_text)
                            } else {
                                content.clone()
                            };

                            let reply_text_content = if ctx.cmd.text_mode && !image_urls.is_empty()
                            {
                                // 使用与 extract_image_urls 相同的逻辑替换
                                let re =
                                    Regex::new(r"!\[.*?\]\(((?:https?://|data:image/)[^\s\)]+)\)")
                                        .unwrap();
                                re.replace_all(content, |caps: &regex::Captures| {
                                    let url = &caps[1];
                                    if url.starts_with("data:") {
                                        "[图片]".to_string()
                                    } else {
                                        url.to_string()
                                    }
                                })
                                .to_string()
                            } else {
                                display_content.clone()
                            };

                            reply(ctx.event, &reply_text_content, ctx.cmd.text_mode, &header).await;

                            for url in &image_urls {
                                if url.starts_with("data:") {
                                    if let Some(base64_data) = url.split(',').nth(1) {
                                        ctx.event.reply(
                                            Message::new()
                                                .add_image(&format!("base64://{}", base64_data)),
                                        );
                                    }
                                } else {
                                    ctx.event.reply(Message::new().add_image(url));
                                }
                            }

                            let video_urls = extract_video_urls(content);
                            for url in video_urls {
                                // 使用 OneBot 标准 video 段发送，data 放 file 字段，框架会自动处理下载/转发
                                let mut vec = Vec::new();
                                let segment = kovi::bot::message::Segment::new(
                                    "video",
                                    kovi::serde_json::json!({
                                        "file": url
                                    }),
                                );
                                vec.push(segment);
                                let msg = kovi::bot::message::Message::from(vec);
                                ctx.event.reply(msg);
                            }
                        }
                    }
                    Err(e) => {
                        {
                            let mut generating = ctx.mgr.generating.write().await;
                            generating.set_generating(ctx.name, is_priv_ctx, &uid, false);
                        }
                        reply_text(ctx.event, format!("❌ API错误: {}", e));
                    }
                },
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
            bot,
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
                    reply_text(event, "🛑 已停止");
                } else {
                    reply_text(event, format!("❌ 智能体 {} 不存在", name));
                }
            }

            Action::Copy => {
                if cmd.args.is_empty() {
                    reply_text(event, "❌ 请指定新名称: 智能体~#新名称");
                    return;
                }

                if cmd.args.chars().count() > 7
                    || cmd.args.chars().any(|c| "&\"#~/ -_'!@$%:*".contains(c))
                {
                    reply_text(event, "❌ 名称限制：最多7字且不能包含指令符号");
                    return;
                }

                let mut c = mgr.config.write().await;
                if c.agents.iter().any(|a| a.name == cmd.args) {
                    reply_text(event, format!("❌ {} 已存在", cmd.args));
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
                    reply_text(event, format!("📑 已复制 {} → {}", name, cmd.args));
                } else {
                    reply_text(event, format!("❌ {} 不存在", name));
                }
            }

            Action::Rename => {
                if cmd.args.is_empty() {
                    reply_text(event, "❌ 请指定新名称: 智能体~=新名称");
                    return;
                }

                if cmd.args.chars().count() > 7
                    || cmd.args.chars().any(|c| "&\"#~/ -_'!@$%:*".contains(c))
                {
                    reply_text(event, "❌ 名称限制：最多7字且不能包含指令符号");
                    return;
                }

                let mut c = mgr.config.write().await;
                if c.agents.iter().any(|a| a.name == cmd.args) {
                    reply_text(event, format!("❌ 目标名称 {} 已存在", cmd.args));
                    return;
                }

                // 先找要重命名的智能体的索引
                let idx_opt = c.agents.iter().position(|a| a.name == *name);
                if let Some(idx) = idx_opt {
                    c.agents[idx].name = cmd.args.clone();
                    mgr.save(&c);
                    reply_text(event, format!("🏷️ 已重命名 {} → {}", name, cmd.args));
                } else {
                    reply_text(event, format!("❌ {} 不存在", name));
                }
            }

            Action::SetDesc => {
                if cmd.args.is_empty() {
                    reply_text(event, "❌ 请提供描述: 智能体:描述内容");
                    return;
                }
                let mut c = mgr.config.write().await;
                if let Some(a) = c.agents.iter_mut().find(|a| a.name == *name) {
                    a.description = cmd.args.clone();
                    mgr.save(&c);
                    reply_text(event, format!("📝 {} 描述已更新", name));
                } else {
                    reply_text(event, format!("❌ {} 不存在", name));
                }
            }

            Action::SetModel => {
                if cmd.args.is_empty() {
                    reply_text(event, "❌ 请指定模型: 智能体%模型名");
                    return;
                }
                let mut c = mgr.config.write().await;
                let models = c.models.clone();
                if let Some(model) = mgr.resolve_model(&cmd.args, &models) {
                    if let Some(a) = c.agents.iter_mut().find(|a| a.name == *name) {
                        let old = a.model.clone();
                        a.model = model.clone();
                        mgr.save(&c);
                        reply_text(event, format!("🔄 {} 模型: {} → {}", name, old, model));
                    } else {
                        reply_text(event, format!("❌ {} 不存在", name));
                    }
                } else {
                    reply_text(event, "❌ 无效模型");
                }
            }

            Action::SetPrompt => {
                let mut c = mgr.config.write().await;
                if let Some(a) = c.agents.iter_mut().find(|a| a.name == *name) {
                    a.system_prompt = cmd.args.clone();
                    mgr.save(&c);
                    if cmd.args.is_empty() {
                        reply_text(event, format!("📝 {} 提示词已清空", name));
                    } else {
                        reply_text(event, format!("📝 {} 提示词已更新", name));
                    }
                } else {
                    reply_text(event, format!("❌ {} 不存在", name));
                }
            }

            Action::ViewPrompt => {
                let c = mgr.config.read().await;
                if let Some(a) = c.agents.iter().find(|a| a.name == *name) {
                    if cmd.text_mode {
                        reply_text(event, &a.system_prompt);
                        return;
                    }
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
                    reply_text(event, format!("❌ {} 不存在", name));
                }
            }

            Action::List => {
                let c = mgr.config.read().await;
                if c.agents.is_empty() {
                    reply_text(event, "📋 暂无智能体，使用 ##名称 模型 提示词 创建");
                    return;
                }

                // 分组逻辑：使用 BTreeMap 自动按模型名称排序
                use std::collections::BTreeMap;
                let mut groups: BTreeMap<String, Vec<(usize, &Agent)>> = BTreeMap::new();

                // 遍历并分组 (保留原始索引 i+1 以便用户操作)
                for (i, a) in c.agents.iter().enumerate() {
                    groups.entry(a.model.clone()).or_default().push((i + 1, a));
                }

                // 生成 HTML
                let mut html_parts = Vec::new();

                // 遍历每一个模型分组
                for (model, mut agents) in groups {
                    // 组内按智能体名称排序
                    agents.sort_by(|a, b| a.1.name.to_lowercase().cmp(&b.1.name.to_lowercase()));

                    // 组头
                    html_parts.push(format!(
                                              r#"<div class="model-group"><div class="model-header"><span>📦 {}</span><span class="model-count">{}</span></div><div class="agent-grid">"#,
                                              model, agents.len()
                                          ));

                    // 组内网格
                    for (real_idx, a) in agents {
                        // 逻辑：优先显示描述；如果没有描述，则截取系统提示词的前 20 个字作为预览；
                        let desc_display = if !a.description.is_empty() {
                            truncate_str(&a.description, 20)
                        } else if !a.system_prompt.is_empty() {
                            truncate_str(&a.system_prompt, 20)
                        } else {
                            "无描述".to_string()
                        };

                        html_parts.push(format!(
                                            r#"<div class="agent-mini"><div class="agent-mini-top"><div class="agent-idx">{}</div><div class="agent-mini-name">{}</div></div><div class="agent-mini-desc">{}</div></div>"#,
                                            real_idx, a.name, desc_display
                                        ));
                    }
                    html_parts.push("</div></div>".to_string());
                }

                let list = html_parts.join("\n");

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
                    reply_text(event, format!("🗑️ 已删除 {}", name));
                } else {
                    reply_text(event, format!("❌ {} 不存在", name));
                }
            }

            Action::ListModels => {
                let c = mgr.config.read().await;

                // 1. 如果配置为空，尝试抓取
                if c.models.is_empty() {
                    drop(c);
                    reply_text(event, "⏳ 正在获取模型列表...");
                    if let Err(e) = mgr.fetch_models().await {
                        reply_text(event, format!("❌ 获取失败: {}", e));
                        return;
                    }
                }

                // 重新读取
                let c = mgr.config.read().await;
                let models = &c.models;

                if models.is_empty() {
                    reply_text(event, "📭 未找到可用模型 (请检查过滤关键字)");
                    return;
                }

                // 2. 统计使用热度 (哪个模型被多少个智能体使用了)
                use std::collections::HashMap;
                let mut usage_count = HashMap::new();
                for agent in &c.agents {
                    *usage_count.entry(agent.model.clone()).or_insert(0) += 1;
                }

                // 3. 动态分组逻辑
                // 直接利用 utils::MODEL_KEYWORDS 进行分组
                let mut groups: HashMap<String, Vec<(usize, String)>> = HashMap::new();
                let mut other_models = Vec::new();

                for (i, m) in models.iter().enumerate() {
                    let idx = i + 1;
                    let lower = m.to_lowercase();
                    let mut matched = false;

                    for &kw in crate::utils::MODEL_KEYWORDS {
                        if lower.contains(kw) {
                            // 将关键字首字母大写作为组名 (e.g. "gpt-5" -> "Gpt-5 Series")
                            let group_name = format!(
                                "{} Series",
                                kw.chars().next().unwrap().to_uppercase().to_string() + &kw[1..]
                            );
                            groups.entry(group_name).or_default().push((idx, m.clone()));
                            matched = true;
                            break;
                        }
                    }

                    if !matched {
                        other_models.push((idx, m.clone()));
                    }
                }

                // 4. 生成 HTML
                let mut html = String::new();

                // 辅助渲染函数
                let render_group = |title: &str, items: &Vec<(usize, String)>| -> String {
                    let mut s = format!(
                        r#"<div class="mod-group"><div class="mod-title">{}</div><div class="chip-box">"#,
                        title
                    );
                    for (idx, name) in items {
                        let badge = if let Some(cnt) = usage_count.get(name) {
                            format!(r#"<span class="chip-bad">{}用</span>"#, cnt)
                        } else {
                            String::new()
                        };
                        s.push_str(&format!(
                                        r#"<div class="chip"><span class="chip-idx">{}</span><span class="chip-name">{}</span>{}</div>"#,
                                        idx, name, badge
                                    ));
                    }
                    s.push_str("</div></div>");
                    s
                };

                // 按 MODEL_KEYWORDS 的定义顺序渲染 (保证顺序可控)
                for &kw in crate::utils::MODEL_KEYWORDS {
                    let group_name = format!(
                        "{} Series",
                        kw.chars().next().unwrap().to_uppercase().to_string() + &kw[1..]
                    );
                    if let Some(items) = groups.get(&group_name) {
                        html.push_str(&render_group(&group_name, items));
                    }
                }

                // 渲染未分类的模型 (如果有漏网之鱼)
                if !other_models.is_empty() {
                    html.push_str(&render_group("Other Models", &other_models));
                }

                // 5. 发送
                reply(
                    event,
                    &html,
                    cmd.text_mode,
                    &format!("🧩 模型列表 (共{}个)", models.len()),
                )
                .await;
            }

            Action::ViewAll(scope) => {
                let c = mgr.config.read().await;
                if let Some(a) = c.agents.iter().find(|a| a.name == *name) {
                    let priv_scope = matches!(scope, Scope::Private);
                    let hist = a.history(priv_scope, &uid);
                    if hist.is_empty() {
                        let s = if priv_scope { "私有" } else { "公有" };
                        reply_text(event, format!("📭 {} {}历史为空", name, s));
                        return;
                    }
                    let content = format_history(hist, 0, cmd.text_mode);
                    let header = format!(
                        "{} {}历史 ({} 条)",
                        name,
                        if priv_scope { "私有" } else { "公有" },
                        hist.len()
                    );
                    reply(event, &content, cmd.text_mode, &header).await;
                } else {
                    reply_text(event, format!("❌ {} 不存在", name));
                }
            }

            Action::ViewAt(scope) => {
                if cmd.indices.is_empty() {
                    reply_text(event, "❌ 请指定索引: 智能体/索引");
                    return;
                }
                let c = mgr.config.read().await;
                if let Some(a) = c.agents.iter().find(|a| a.name == *name) {
                    let priv_scope = matches!(scope, Scope::Private);
                    let hist = a.history(priv_scope, &uid);
                    let mut results = Vec::new();
                    let mut extra_images = Vec::new();

                    let re =
                        Regex::new(r"!\[.*?\]\(((?:https?://|data:image/)[^\s\)]+)\)").unwrap();

                    for i in &cmd.indices {
                        if *i > 0 && *i <= hist.len() {
                            let m = &hist[i - 1];
                            let emoji = match m.role.as_str() {
                                "user" => "👤",
                                "assistant" => "🤖",
                                _ => "❓",
                            };

                            let mut content = m.content.clone();
                            let mut msg_imgs = extract_image_urls(&content);
                            msg_imgs.extend(m.images.clone());

                            if cmd.text_mode {
                                content = re
                                    .replace_all(&content, |caps: &regex::Captures| {
                                        let url = &caps[1];
                                        if url.starts_with("data:") {
                                            "[图片]".to_string()
                                        } else {
                                            url.to_string()
                                        }
                                    })
                                    .to_string();
                            }

                            if !m.images.is_empty() {
                                if !content.is_empty() {
                                    content.push_str("\n\n");
                                }
                                for url in &m.images {
                                    if cmd.text_mode {
                                        if url.starts_with("data:") {
                                            content.push_str("\n- [Base64 Image]");
                                        } else {
                                            content.push_str(&format!("\n- {}", url));
                                        }
                                    } else {
                                        content.push_str(&format!("\n![image]({})", url));
                                    }
                                }
                            }

                            extra_images.extend(msg_imgs);

                            results.push(format!("**#{} {}**\n{}", i, emoji, content));
                        }
                    }

                    if results.is_empty() {
                        reply_text(event, "❌ 索引无效");
                    } else {
                        reply(
                            event,
                            &results.join("\n\n---\n\n"),
                            cmd.text_mode,
                            &format!("{} 历史记录", name),
                        )
                        .await;

                        for url in extra_images {
                            if url.starts_with("data:") {
                                if let Some(base64_data) = url.split(',').nth(1) {
                                    event.reply(
                                        Message::new()
                                            .add_image(&format!("base64://{}", base64_data)),
                                    );
                                }
                            } else {
                                event.reply(Message::new().add_image(&url));
                            }
                        }
                    }
                } else {
                    reply_text(event, format!("❌ {} 不存在", name));
                }
            }

            Action::Export(scope) => {
                let c = mgr.config.read().await;
                if let Some(a) = c.agents.iter().find(|a| a.name == *name) {
                    let priv_scope = matches!(scope, Scope::Private);
                    let hist = a.history(priv_scope, &uid);
                    if hist.is_empty() {
                        reply_text(event, "📭 历史为空");
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
                                    Ok(_) => reply_text(event, format!("📤 已导出: {}", fname)),
                                    Err(e) => reply_text(event, format!("❌ 上传失败: {}", e)),
                                }
                            } else {
                                reply_text(event, "❌ 写入失败");
                            }
                        }
                        Err(e) => reply_text(event, format!("❌ 创建文件失败: {}", e)),
                    }
                } else {
                    reply_text(event, format!("❌ {} 不存在", name));
                }
            }

            Action::EditAt(scope) => {
                if cmd.indices.is_empty() {
                    reply_text(event, "❌ 请指定索引: 智能体'索引 新内容");
                    return;
                }
                if cmd.args.is_empty() {
                    reply_text(event, "❌ 请提供新内容");
                    return;
                }
                let idx = cmd.indices[0];
                let mut c = mgr.config.write().await;
                if let Some(a) = c.agents.iter_mut().find(|a| a.name == *name) {
                    let priv_scope = matches!(scope, Scope::Private);
                    if a.edit_at(priv_scope, &uid, idx, &cmd.args) {
                        mgr.save(&c);
                        reply_text(event, format!("✏️ 已编辑第 {} 条", idx));
                    } else {
                        reply_text(event, format!("❌ 索引 {} 无效", idx));
                    }
                } else {
                    reply_text(event, format!("❌ {} 不存在", name));
                }
            }

            Action::DeleteAt(scope) => {
                if cmd.indices.is_empty() {
                    reply_text(event, "❌ 请指定索引: 智能体-索引 (支持 1,3,5 或 1-5)");
                    return;
                }
                let mut c = mgr.config.write().await;
                if let Some(a) = c.agents.iter_mut().find(|a| a.name == *name) {
                    let priv_scope = matches!(scope, Scope::Private);
                    let deleted = a.delete_at(priv_scope, &uid, &cmd.indices);
                    if deleted.is_empty() {
                        reply_text(event, "❌ 索引无效");
                    } else {
                        mgr.save(&c);
                        let s = deleted
                            .iter()
                            .map(|i| i.to_string())
                            .collect::<Vec<_>>()
                            .join(", ");
                        reply_text(
                            event,
                            format!("🗑️ 已删除第 {} 条 (共{}条)", s, deleted.len()),
                        );
                    }
                } else {
                    reply_text(event, format!("❌ {} 不存在", name));
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
                    reply_text(event, format!("🧹 {} {}历史已清空", name, s));
                } else {
                    reply_text(event, format!("❌ {} 不存在", name));
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
                reply_text(event, format!("🧹 已清空 {} 个智能体的公有历史", cnt));
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
                reply_text(event, format!("⚠️ 已清空 {} 个智能体的所有历史", cnt));
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
| `##名称 模型 提示词` | 创建/更新 | `##助手 gpt-4o 你是助手` |
| `##:模型` | 批量生成描述 | `##:gpt-4o` |
| `智能体~=新名` | 重命名 | `助手~=管家` |
| `智能体~#新名` | 复制 | `助手~#助手2` |
| `智能体:描述` | 设置描述 | `助手:通用助手` |
| `-#名称` | 删除 | `-#助手` |
| `/#` | 列表 | `/#` |

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

            Action::AutoFillDescriptions(model_ref) => {
                let (target_agents, api_config, use_model) = {
                    let c = mgr.config.read().await;

                    // 1. 确定使用的模型
                    let models = c.models.clone();
                    let resolved_model = if model_ref.is_empty() {
                        c.default_model.clone()
                    } else {
                        mgr.resolve_model(&model_ref, &models).unwrap_or(model_ref)
                    };

                    // 2. 筛选需要生成的智能体 (描述为空 或 仅仅是"新建智能体")
                    let targets: Vec<(String, String)> = c
                        .agents
                        .iter()
                        .filter(|a| a.description.is_empty() || a.description == "新建智能体")
                        .map(|a| (a.name.clone(), a.system_prompt.clone()))
                        .collect();

                    (
                        targets,
                        (c.api_base.clone(), c.api_key.clone()),
                        resolved_model,
                    )
                };

                if target_agents.is_empty() {
                    reply_text(event, "✅ 所有智能体均已有描述，无需处理。");
                    return;
                }

                if api_config.0.is_empty() || api_config.1.is_empty() {
                    reply_text(event, "❌ API 未配置");
                    return;
                }

                reply_text(
                    event,
                    format!(
                        "🤖 开始使用 [{}] 为 {} 个智能体生成描述，请稍候...",
                        use_model,
                        target_agents.len()
                    ),
                );

                let client = Client::with_config(
                    OpenAIConfig::new()
                        .with_api_base(api_config.0)
                        .with_api_key(api_config.1),
                );

                let mut success_count = 0;

                for (name, prompt) in target_agents {
                    // 这里的 Prompt 专门用于生成简短描述
                    let gen_prompt = format!(
                        "请阅读以下角色的 System Prompt，为其生成一个极简短的中文功能描述（Role/Tag）。\n\
                                    要求：\n1. 必须控制在 10 个字以内\n2. 不要包含任何标点符号\n3. 直接输出描述内容，不要解释\n\n\
                                    System Prompt:\n{}",
                        prompt
                    );

                    let req = CreateChatCompletionRequestArgs::default()
                        .model(&use_model)
                        .messages(vec![
                            ChatCompletionRequestUserMessageArgs::default()
                                .content(gen_prompt)
                                .build()
                                .unwrap()
                                .into(),
                        ])
                        .build();

                    if let Ok(req) = req
                        && let Ok(res) = client.chat().create(req).await
                        && let Some(choice) = res.choices.first()
                        && let Some(content) = &choice.message.content
                    {
                        let new_desc = content.trim().replace(['"', '“', '”', '。', '.'], ""); // 简单清洗

                        // 获取写锁更新数据
                        let mut c = mgr.config.write().await;
                        if let Some(a) = c.agents.iter_mut().find(|a| a.name == name) {
                            a.description = new_desc.clone();
                            mgr.save(&c);
                            success_count += 1;
                        }
                    }

                    // 小停顿，避免并发过高 (100毫秒)
                    kovi::tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                }

                reply_text(
                    event,
                    format!("✅ 批量处理完成，已更新 {} 个智能体的描述。", success_count),
                );
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
            reply_text(
                event,
                format!("📝 已更新 {} (模型: {})", name, updated_model),
            );
        } else {
            let description = if desc.is_empty() {
                "新建智能体".to_string()
            } else {
                desc.to_string()
            };
            c.agents
                .push(Agent::new(name, &model, &prompt, &description));
            mgr.save(&c);
            reply_text(event, format!("🤖 已创建 {} (模型: {})", name, model));
        }
    }
}

// --- 入口 ---
use crate::logic::reply_text;
use cdp_html_shot::Browser;
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
                reply_text(&event, format!("✅ API 已配置: {}", url));
                match mgr.fetch_models().await {
                    Ok(models) => reply_text(&event, format!("📋 已获取 {} 个模型", models.len())),
                    Err(e) => reply_text(&event, format!("⚠️ 获取模型失败: {}", e)),
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
                let (quote, imgs) = utils::get_full_content(&event, &bot).await;

                // 拼接提示词：引用 + 用户输入参数
                let prompt = if matches!(
                    cmd.action,
                    parser::Action::Chat | parser::Action::Regenerate
                ) {
                    format!("{}{}", quote, cmd.args).trim().to_string()
                } else {
                    cmd.args.clone()
                };

                logic::execute(cmd, prompt, imgs, &event, &mgr, &bot).await;
            }
        }
    });

    let mgr_drop = mgr.clone();
    PluginBuilder::drop({
        move || {
            let mgr = mgr_drop.clone();
            async move {
                // 保存配置
                let c = mgr.config.read().await;
                mgr.save(&c);
                // 关闭全局浏览器实例
                // Browser::instance().await.close_async().await.unwrap();
                cdp_html_shot::Browser::shutdown_global().await;
            }
        }
    });
}
