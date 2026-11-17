# 联网搜索插件

`web_search` 仍是原来的 function tool。用户问到实时信息时，主对话模型像调用查天气一样调用它。插件再单独请求 DeepSeek 官方搜索接口，把标题、链接和片段交回同一次对话。

不要把 DeepSeek 的服务端搜索工具和现有 function tools 放进同一次对话请求。这两种混在一起时，接口会返回 400。

## 请求

- 地址固定为 `POST https://api.deepseek.com/v1/tools/search`。不读取插件里残留的 `base_url`，避免把密钥发到别的地址。遇到跳转会中止，不跟随。
- 认证与聊天相同：`Authorization: Bearer <DeepSeek API Key>`。
- 请求体只有 `query`（1–128 个字符）、`count`（这里默认 5，最多取 10）、`need_content: false`。不要正文，减少把整页内容当成指令的机会。
- 成功时响应顶层是 `results`。每条使用标题、链接、片段；片段为空时才用截短的摘要。`content`、站点图标、`usage`、`citations` 不交给模型。
- 空的 `results` 表示这次没找到网页，不是失败，也不要编造成已经搜到。
- 非空的成功请求才计费。本次部署说明是每 1000 次 5 元；官方文档当前写的是每 1000 次 0.75 美元。空结果、失败和超时不计费。后台「测试」会真实搜一次。

密钥按这个顺序取，都不写入仓库、日志或交回模型的文字：

1. 这个连接正在用的模型，前提是条目名叫 `DeepSeekLLM`，或接口主机是 `api.deepseek.com` / `*.deepseek.com`，并且 Key 不是占位文字。
2. 配置里选定的大模型，条件同上。
3. `plugins.web_search.api_key` 备用。留空表示不用。

带 `provider`（秘塔、Tavily、Serply）的旧 Key，以及 `mk-`、`tvly-` 开头的 Key，不会发到 DeepSeek。

网页片段按不可信数据处理：压成一行、截断，并标明不是指令。模型不应执行片段里的要求，也不能把每一条当成已经核实的事实。

整次搜索最多等 20 秒（连接最多 5 秒）。开始前先说「我上网查一下。」没有 Key 或搜索词不合规时不说这句，也不会发出请求。

## 配置

主对话模型不用改。选定 `DeepSeekLLM` 并填好它自己的 Key 后，搜索就能用。

备用 Key 和条数可以写在 `data/.config.yaml`：

```yaml
plugins:
  web_search:
    api_key: ""          # 留空：用上面选定的 DeepSeek Key
    count: 5             # 1 到 10
Intent:
  function_call:
    functions:
      - web_search
```

管理后台「意图识别」页勾选「联网搜索」。旁边的「设置」里，备用 Key 只写不读；「测试」留空 Key 时，和正式调用一样解析选定模型的 Key。

## 和办公室已装版本的关系

已装 Hub 若停在 `a6e9a8300025b5a2efd75dd4d6a37f36c55dab79`，当时的 `web_search` 调用秘塔、Tavily 或 Serply。只替换 `main/xiaozhi-server/plugins_func/functions/web_search.py` 即可改到上述行为：函数名、`async`、`Action.REQLLM` 的返回方式不变，不依赖之后才加入的后台设置模块。

这个文件不会改主对话模型、思考开关、其它工具、微信或摄像头。设备上的 `data/.config.yaml` 不在本仓库里。启用前需要核对的非密钥事实：选定条目是不是 `DeepSeekLLM`（或主机是不是 `api.deepseek.com`）、`model_name` 是什么、`plugins.web_search.provider` 是否还在。密钥本身不要发出来。
