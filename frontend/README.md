# Agent Workbench Frontend

面向智能客服 Agent 的开发工作台。页面默认连接真实 FastAPI 后端，不包含 Mock 数据分支。

## 本地运行

先在项目根目录启动后端：

```powershell
D:\python\agentProject\.venv\Scripts\python.exe main.py
```

再启动前端：

```powershell
cd frontend
npm install
npm run dev
```

默认使用 `/api`，Vite 会将请求代理到 `http://127.0.0.1:8000`。如需连接其他服务，可在 `.env.local` 中配置：

```env
VITE_API_BASE_URL=http://127.0.0.1:8000/api
```

## 已连接接口

- Agent 与工具：`GET /api/agents`、`GET /api/tools`
- 会话：`GET/POST /api/sessions`、`PATCH/DELETE /api/sessions/:id`
- 历史消息：`GET /api/history/:sessionId`
- 工作区文件：`GET /api/files`
- 对话：`POST /api/chat`、`POST /api/chat_stream`
- 运行控制：`POST /api/agent/stop`、`GET /api/agent/status/:sessionId`

## 验证

```powershell
npm run typecheck
npm run lint
npm run build
```
