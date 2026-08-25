import axios, { AxiosError } from "axios"

export interface ApiErrorPayload { detail?: string; message?: string }

export const apiClient = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL ?? "/api",
  timeout: 30_000,
  headers: { "Content-Type": "application/json", Accept: "application/json" },
})

apiClient.interceptors.request.use((config) => {
  config.headers.set("X-Client", "agent-workbench")
  config.headers.set("X-Request-ID", crypto.randomUUID())
  return config
})

apiClient.interceptors.response.use(
  (response) => response,
  (error: AxiosError<ApiErrorPayload>) => {
    const message = error.response?.data.detail ?? error.response?.data.message ?? error.message ?? "请求失败"
    return Promise.reject(new Error(message))
  },
)
