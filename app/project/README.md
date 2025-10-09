# 小蓝猫（Bluecat）心理健康助手

本仓库为开源版本，当前功能以 App 底部导航栏的四个模块为准：AI心理咨询、治疗建议、测试、我的。

## 功能概览
- AI心理咨询：
  - 端侧 LLaMA 推理（C++ 原生库）保护隐私
  - 多轮智能对话、聊天记录本地保存
  - 网络安全配置支持开发环境（HTTP）与模拟器域名
- 治疗建议：
  - 记录并管理个性化治疗建议
  - 支持新增、编辑、删除与列表查看
  - 与用户信息关联，便于后续复盘
- 测试：
  - 支持 SCL-90、SAS、SDS、MBTI 等常见心理评估
  - 生成测试报告与建议，支持历史结果查看
  - 结果本地存储，服务端接口可选对接（见下文后端）
- 我的：
  - 用户注册/登录（JWT）、头像上传、个人信息编辑
  - 密码修改、退出登录等账号管理

## 技术栈
- 前端（Android App）
  - 语言：Java（少量 Kotlin 协程）
  - 网络：Retrofit2 + OkHttp3 + Gson
  - 图片：Glide
  - 原生：C++17（llama.cpp，通过 CMake/NDK 构建）
  - 本地存储：SQLite
- 后端（可选对接）
  - Spring Boot + Spring Security + JWT
  - MyBatis-Plus，MySQL 8.0
  - 说明文件：`server/src/main/resources/application.yml.example`

## 项目结构
- `app/` Android 前端源码与资源（`src/main/java`、`src/main/res`、`src/main/cpp`）
- `server/` Java 后端（`pom.xml`、`src/main/java`、`src/main/resources` 示例配置）
- 构建与工具：`build.gradle`、`settings.gradle`、`gradle.properties`、`gradlew*`、`gradle/wrapper/`
- 开源协议：`LICENSE`

## 快速开始
- 前置依赖
  - Android Studio（含 NDK/CMake）、JDK 8+、MySQL（如需对接后端）
- 后端（可选）
  - 将 `application.yml.example` 复制为 `application.yml` 并填写：
    - `spring.datasource.url/username/password`
    - `jwt.secret`（自定义密钥）
  - 使用 Maven 构建运行：`mvn package` 或 `mvn spring-boot:run`
  - 默认端口 `8080`
- Android
  - 在 `app/src/main/java/com/example/project/api/ApiClient.java` 将 `BASE_URL` 指向后端地址（示例：`http://localhost:8080/`）
  - 打开项目等待 Gradle 同步，连接设备或启动模拟器后运行
  - 网络安全：`app/src/main/res/xml/network_security_config.xml` 已包含 `localhost/127.0.0.1/10.0.2.2` 及局域网占位

## 模型与隐私
- 模型文件不随仓库发布。
- 请勿提交密钥、私有配置或用户数据；本仓库 `.gitignore` 已忽略常见敏感文件（如 `application.yml`、签名文件、构建产物等）。

## 贡献
- 欢迎提交 Issue 与 Pull Request 来改进代码质量、完善模块与文档。
- 提交前请确保通过本地构建与基本功能验证。

## 开发人员
- 作者：zane Wang
- 负责范围：整体架构设计、Android 客户端、后端服务、文档与开源维护
- 联系方式：email: 2640289029@qq.com  github: https://github.com/zaneWWWWWW

## 许可证
- 本项目采用 MIT 许可证，详见根目录 `LICENSE` 文件。