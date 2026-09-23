# Smart CS Agent V3

电信与零售复合场景智能客服项目，前端使用 React/Vite，后端使用 FastAPI/LangGraph。

## Docker 一键启动

Windows：

```powershell
.\deploy.ps1
```

Linux / macOS：

```bash
chmod +x deploy.sh
./deploy.sh up
```

首次启动会安全地生成本地部署密钥并提示输入 Qwen API Key。服务健康后访问 <http://localhost>。

完整配置、演示数据、升级和运维说明见 [DOCKER_DEPLOY.md](DOCKER_DEPLOY.md)。

