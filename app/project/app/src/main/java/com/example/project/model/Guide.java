package com.example.project.model;

public class Guide {
    private long id;
    private String content;
    private boolean isDefault; // true表示系统默认指南，false表示用户自定义指南
    private boolean isCompleted; // true表示已完成，false表示未完成
    private long userId; // 关联的用户ID，系统默认指南的userId为0
    private long timestamp; // 创建时间戳

    public Guide() {
    }

    public Guide(String content, boolean isDefault, long userId) {
        this.content = content;
        this.isDefault = isDefault;
        this.isCompleted = false;
        this.userId = userId;
        this.timestamp = System.currentTimeMillis();
    }

    public long getId() {
        return id;
    }

    public void setId(long id) {
        this.id = id;
    }

    public String getContent() {
        return content;
    }

    public void setContent(String content) {
        this.content = content;
    }

    public boolean isDefault() {
        return isDefault;
    }

    public void setDefault(boolean aDefault) {
        isDefault = aDefault;
    }

    public boolean isCompleted() {
        return isCompleted;
    }

    public void setCompleted(boolean completed) {
        isCompleted = completed;
    }

    public long getUserId() {
        return userId;
    }

    public void setUserId(long userId) {
        this.userId = userId;
    }

    public long getTimestamp() {
        return timestamp;
    }

    public void setTimestamp(long timestamp) {
        this.timestamp = timestamp;
    }
}