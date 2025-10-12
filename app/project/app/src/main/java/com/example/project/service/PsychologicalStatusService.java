package com.example.project.service;

import android.content.Context;
import android.util.Log;

import com.example.project.LLamaAPI;
import com.example.project.db.ChatDbHelper;
import com.example.project.model.Message;

import java.io.File;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * 心理状态评估服务
 * 负责调用量化后的模型进行心理状态评估
 */
public class PsychologicalStatusService {
    private static final String TAG = "PsychologicalStatusService";
    private static final String MODEL_NAME = "Bluecat_status.gguf";
    
    private final Context context;
    private final ChatDbHelper dbHelper;
    private final ExecutorService executorService;
    private LLamaAPI statusLlamaAPI; // 专用于状态判断的LLamaAPI实例
    private boolean isModelLoaded = false;
    private boolean isExternalModelInstance = false; // 标记是否使用外部提供的模型实例
    private com.example.project.utils.PsychologicalStatusManager statusManager;
    
    public interface AnalysisCallback {
        void onSuccess(String analysisResult);
        void onError(Exception e);
    }
    
    public PsychologicalStatusService(Context context) {
        this.context = context;
        this.dbHelper = new ChatDbHelper(context);
        this.executorService = Executors.newSingleThreadExecutor();
        this.statusManager = new com.example.project.utils.PsychologicalStatusManager(context);
        // 不在构造函数中初始化模型，避免在Fragment创建时就加载/卸载模型
        // 将在需要分析时才初始化模型
    }
    
    /**
     * 设置外部提供的状态判断模型实例
     * @param statusLlamaAPI 外部提供的状态判断模型实例
     */
    public void setStatusLlamaAPI(LLamaAPI statusLlamaAPI) {
        if (statusLlamaAPI != null) {
            this.statusLlamaAPI = statusLlamaAPI;
            this.isExternalModelInstance = true;
            Log.d(TAG, "使用外部提供的状态判断模型实例");
            
            // 检查模型是否已加载
            this.isModelLoaded = statusLlamaAPI.isModelLoaded();
            Log.d(TAG, "外部状态判断模型已加载: " + this.isModelLoaded);
        }
    }
    
    // 移除initializeModel方法，直接在analyzeUserPsychologicalStatus中处理模型加载
    
    /**
     * 获取状态判断模型的文件路径
     * @return 状态判断模型的文件路径
     */
    private String getStatusModelPath() {
        File filesDir = context.getFilesDir();
        File modelsDir = new File(filesDir, "models");
        if (!modelsDir.exists()) {
            boolean created = modelsDir.mkdirs();
            Log.d(TAG, "创建模型目录: " + created);
        }
        File modelFile = new File(modelsDir, MODEL_NAME);
        String path = modelFile.getAbsolutePath();
        Log.d(TAG, "状态模型路径: " + path);
        Log.d(TAG, "状态模型文件存在: " + modelFile.exists());
        if (modelFile.exists()) {
            Log.d(TAG, "状态模型文件大小: " + modelFile.length() + " bytes");
            Log.d(TAG, "状态模型文件可读: " + modelFile.canRead());
        }
        return path;
    }
    
    /**
     * 分析用户的心理状态
     * @param callback 分析结果回调
     */
    public void analyzeUserPsychologicalStatus(AnalysisCallback callback) {
        if (callback == null) {
            throw new IllegalArgumentException("回调不能为空");
        }
        
        executorService.execute(() -> {
            try {
                // 记录开始时间
                long startTime = System.currentTimeMillis();
                Log.d(TAG, "开始心理状态分析...");
                
                // 如果没有外部提供的模型实例，则创建新的实例
                if (statusLlamaAPI == null && !isExternalModelInstance) {
                    statusLlamaAPI = LLamaAPI.createInstance("status_model");
                    Log.d(TAG, "创建状态判断专用LLamaAPI实例");
                } else if (statusLlamaAPI == null) {
                    callback.onError(new IllegalStateException("状态判断模型实例未设置"));
                    return;
                }
                
                // 从数据库获取最近的五条用户消息（不包括AI回复）
                List<Message> messages = dbHelper.getRecentUserMessages(5);
                Log.d(TAG, "获取最近5条用户消息进行心理状态分析，不包括AI回复");
                
                // 构建提示词
                StringBuilder prompt = new StringBuilder();
                prompt.append("你是一个专业的心理咨询师，请根据以下用户消息内容，分析用户可能的心理状态、情绪和潜在的心理健康问题。\n\n");
                prompt.append("用户消息历史：\n");
                
                // 添加最近的用户消息记录
                for (Message message : messages) {
                    prompt.append("用户: ").append(message.getContent()).append("\n");
                }
                
                // 记录用户消息数量
                Log.d(TAG, "用于分析的用户消息数量: " + messages.size());
                
                prompt.append("\n请提供专业的心理状态评估报告，包括：\n");
                prompt.append("1. 用户当前的情绪状态（必须明确标注为：正常、一般或较差）\n");
                prompt.append("2. 可能存在的心理健康问题或风险\n");
                prompt.append("3. 建议的沟通方式和支持策略\n");
                prompt.append("\n请以JSON格式返回抑郁和焦虑评分，格式为：{\"depression\": 0-3的整数, \"anxiety\": 0-3的整数}\n");
                prompt.append("其中0表示无，1表示轻度，2表示中度，3表示重度\n");
                
                // 记录提示词
                Log.d(TAG, "心理状态分析提示词: " + prompt.toString());
                
                // 检查状态判断模型是否已加载，如果没有则加载
                String statusModelPath = getStatusModelPath();
                boolean needToLoadStatusModel = false;
                
                if (!statusLlamaAPI.isModelLoaded()) {
                    // 如果没有模型加载，直接加载状态判断模型
                    Log.d(TAG, "状态判断实例没有模型加载，加载状态判断模型: " + statusModelPath);
                    needToLoadStatusModel = true;
                } else {
                    // 记录当前使用的模型名称
                    String currentModelName = statusLlamaAPI.getCurrentModelName();
                    // 如果当前加载的不是状态判断模型，则需要切换
                    if (currentModelName == null || !currentModelName.contains("Bluecat_status")) {
                        Log.d(TAG, "状态判断实例当前模型不是状态判断模型，需要切换: 当前 = " + currentModelName + ", 目标 = " + MODEL_NAME);
                        // 卸载当前模型
                        statusLlamaAPI.unloadModel();
                        needToLoadStatusModel = true;
                    } else {
                        Log.d(TAG, "状态判断实例当前已加载状态判断模型，直接使用: " + currentModelName);
                    }
                }
                
                // 如果需要，加载状态判断模型
                if (needToLoadStatusModel) {
                    try {
                        // 获取状态判断模型路径
                        String modelPath = getStatusModelPath();
                        File modelFile = new File(modelPath);
                        if (!modelFile.exists()) {
                            throw new IllegalStateException("状态判断模型文件不存在: " + modelPath);
                        }
                        if (!modelFile.canRead()) {
                            throw new IllegalStateException("状态判断模型文件无法读取: " + modelPath);
                        }
                        
                        Log.d(TAG, "加载状态判断模型: " + modelPath);
                        statusLlamaAPI.loadModel(modelPath);
                        
                        // 等待模型加载完成
                        Thread.sleep(3000); // 增加初始等待时间到3秒
                        
                        // 验证模型是否正确加载
                        int maxRetries = 10; // 增加重试次数到10次
                        boolean modelLoaded = false;
                        
                        for (int i = 0; i < maxRetries && !modelLoaded; i++) {
                            modelLoaded = statusLlamaAPI.isModelLoaded();
                            Log.d(TAG, "检查状态判断模型加载状态 (尝试 " + (i+1) + "/" + maxRetries + "): " + modelLoaded);
                            
                            if (!modelLoaded) {
                                // 使用指数退避策略增加等待时间
                                int waitTime = 1000 * (i + 1); // 从1秒开始，逐渐增加
                                Log.d(TAG, "等待" + waitTime + "毫秒后重试");
                                Thread.sleep(waitTime); // 增加每次等待时间
                            }
                        }
                        
                        // 最终验证
                        if (!statusLlamaAPI.isModelLoaded()) {
                            throw new IllegalStateException("状态判断模型加载失败");
                        }
                        
                        // 验证加载的是正确的模型
                        String loadedModelName = statusLlamaAPI.getCurrentModelName();
                        if (loadedModelName == null || !loadedModelName.equals(MODEL_NAME)) {
                            Log.e(TAG, "加载的模型名称不匹配: 期望=" + MODEL_NAME + ", 实际=" + loadedModelName);
                            throw new IllegalStateException("加载了错误的模型: " + loadedModelName);
                        }
                        
                        isModelLoaded = true;
                        Log.d(TAG, "状态判断模型加载成功");
                    } catch (Exception e) {
                        Log.e(TAG, "加载状态判断模型失败", e);
                        callback.onError(new Exception("加载状态判断模型失败: " + e.getMessage(), e));
                        return;
                    }
                }
                
                // 再次验证模型是否已加载
                if (!statusLlamaAPI.isModelLoaded()) {
                    Log.e(TAG, "状态判断模型未加载，无法进行分析");
                    if (callback != null) {
                        // 返回默认的心理状态评估结果（轻度抑郁和焦虑）
                        Log.w(TAG, "模型未加载，返回默认心理状态评估结果");
                        callback.onSuccess("{\"depression\": 1, \"anxiety\": 1}");
                    }
                    return;
                }
                
                // 记录模型加载完成时间
                long modelLoadTime = System.currentTimeMillis();
                Log.d(TAG, "模型加载完成，耗时: " + (modelLoadTime - startTime) + "ms");
                
                // 生成分析结果
                final String[] result = {""};
                final String currentModelName = statusLlamaAPI.getCurrentModelName();
                Log.d(TAG, "使用状态判断模型进行心理状态分析: " + currentModelName);
                
                // 再次确认模型已加载并准备好
                if (!statusLlamaAPI.isModelLoaded() || currentModelName == null || !currentModelName.contains("Bluecat_status")) {
                    Log.e(TAG, "状态判断模型未正确加载: " + currentModelName);
                    if (callback != null) {
                        // 返回默认的心理状态评估结果（轻度抑郁和焦虑）
                        Log.w(TAG, "模型未正确加载，返回默认心理状态评估结果");
                        callback.onSuccess("{\"depression\": 1, \"anxiety\": 1}");
                    }
                    return;
                }
                // 记录推理开始时间
                final long inferenceStartTime = System.currentTimeMillis();
                statusLlamaAPI.chat(prompt.toString(), new LLamaAPI.CompletionCallback() {
                    @Override
                    public void onToken(String token) {
                        result[0] += token;
                    }
                    
                    @Override
                    public void onComplete() {
                        // 记录推理完成时间
                        long inferenceEndTime = System.currentTimeMillis();
                        Log.d(TAG, "心理状态分析完成，推理耗时: " + (inferenceEndTime - inferenceStartTime) + "ms");
                        Log.d(TAG, "心理状态分析结果: " + result[0]);
                        
                        // 使用独立实例，不需要恢复原来的模型
                        
                        // 只记录推理耗时，不统计模型切换和加载时间
                        // long endTime = System.currentTimeMillis();
                        // Log.d(TAG, "心理状态分析总耗时: " + (endTime - startTime) + "ms");
                        
                        // 保存分析结果到本地
                        statusManager.saveStatusResult(result[0]);
                        // 在日志中显示完整的评估结果历史JSON字符串
                        String historyJson = statusManager.getStatusHistory();
                        Log.d(TAG, "心理状态评估历史记录: " + historyJson);
                        
                        // 返回分析结果
                        if (callback != null) {
                            callback.onSuccess(result[0]);
                        }
                    }
                    
                    @Override
                    public void onError(Exception e) {
                        Log.e(TAG, "生成分析结果失败", e);
                        
                        // 使用独立实例，不需要恢复原来的模型
                        
                        if (callback != null) {
                            callback.onError(e);
                        }
                    }
                });
                
            } catch (Exception e) {
                Log.e(TAG, "执行心理状态分析失败", e);
                if (callback != null) {
                    // 在模型加载失败时返回默认值
                    if (e.getMessage() != null && (e.getMessage().contains("模型加载失败") || e.getMessage().contains("模型未加载"))) {
                        Log.w(TAG, "模型加载失败，返回默认心理状态评估结果");
                        // 返回默认的心理状态评估结果（轻度抑郁和焦虑）
                        callback.onSuccess("{\"depression\": 1, \"anxiety\": 1}");
                    } else {
                        callback.onError(e);
                    }
                }
            }
        });
    }
    
    /**
     * 释放资源
     */
    public void release() {
        executorService.shutdown();
        // 关闭数据库连接
        if (dbHelper != null) {
            dbHelper.close();
        }
    }
}