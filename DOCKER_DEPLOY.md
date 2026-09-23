# Docker 一键部署

该部署包含四个容器：

- `frontend`：Nginx 托管 Vite 构建产物，并代理 `/api`、SSE 和健康检查。
- `backend`：FastAPI 应用，启动时自动执行 `backend/migrations` 中尚未应用的迁移。
- `mysql`：业务事实、会话、治理动作和检查点的持久化数据库。
- `redis`：近期消息缓存；Redis 异常不会替代 MySQL 事实源。

只有 Nginx 的 HTTP 端口会映射到宿主机。MySQL、Redis 和后端端口不会公开。

## 前置条件

- Docker Engine 或 Docker Desktop
- Docker Compose v2
- 可用的 `QWEN_API_KEY`

## Windows 一键启动

在项目根目录运行：

```powershell
.\deploy.ps1
```

也可以双击 `deploy.bat`。首次启动会：

1. 从 `.env.docker.example` 创建不会入库的 `.env.docker`。
2. 自动生成 MySQL、PII 加密和观测哈希密钥。
3. 隐藏输入并保存 `QWEN_API_KEY`，也可预先通过环境变量提供。
4. 构建镜像、启动服务并等待全部健康检查通过。

如需同时写入演示业务数据：

```powershell
.\deploy.ps1 -Action up -Seed
```

## Linux / macOS 一键启动

```bash
chmod +x deploy.sh
./deploy.sh up
```

写入演示数据：

```bash
./deploy.sh up --seed
```

默认访问地址为 <http://localhost>。如端口 80 已被占用，先修改 `.env.docker`：

```env
HTTP_PORT=8080
```

然后访问 `http://localhost:8080`。

## 常用运维命令

```powershell
.\deploy.ps1 -Action status   # 查看服务与健康状态
.\deploy.ps1 -Action logs     # 跟踪所有容器日志
.\deploy.ps1 -Action restart  # 重新构建并滚动到最新代码
.\deploy.ps1 -Action down     # 停止容器，保留数据卷
```

Linux/macOS 将上述命令对应替换为 `./deploy.sh status|logs|restart|down`。

直接使用 Compose 也可以：

```bash
docker compose --env-file .env.docker up -d --build --wait
docker compose --env-file .env.docker logs -f backend
docker compose --env-file .env.docker down
```

健康检查：

- 存活检查：`GET /health/live`
- 就绪检查：`GET /health/ready`

## 数据与升级

`mysql_data`、`redis_data` 和 `backend_vector_data` 都是具名数据卷。普通 `down`、重新构建镜像或升级代码不会删除它们。

后端启动时会校验已应用迁移的 SHA-256；不要修改已经部署过的 SQL 文件，应新增递增编号的迁移。升级代码后执行 `restart` 即可。

只有在明确要清空全部业务数据时，才执行下面的不可恢复命令：

```bash
docker compose --env-file .env.docker down --volumes
```

## 生产环境注意事项

- `.env.docker` 已被 Git 忽略，不要将其复制到镜像或提交到仓库。
- 当前 Nginx 会主动删除外部传入的 `X-User-Id`，防止伪造可信身份头。正式启用身份头前，应先接入认证网关，并同步配置可信代理网段。
- 公网部署应在 Nginx 前或 Nginx 中启用 HTTPS、访问控制、证书自动续期和限流。
- Qdrant 和 Langfuse 默认关闭；需要时在 `.env.docker` 中配置外部服务。它们不是默认启动链路的必要条件。
- 备份 MySQL 数据卷后再做版本升级或破坏性运维。

