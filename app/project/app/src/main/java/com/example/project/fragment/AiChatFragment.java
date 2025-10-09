package com.example.project.fragment;

import android.app.AlertDialog;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.EditText;
import android.widget.ImageButton;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import com.example.project.LLamaAPI;
import com.example.project.R;
import com.example.project.adapter.MessageAdapter;
import com.example.project.db.ChatDbHelper;
import com.example.project.model.Message;
import com.example.project.utils.ConversationCounter;
import com.example.project.service.PsychologicalStatusService;
import com.example.project.utils.ModelLoadingDialogManager;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

public class AiChatFragment extends Fragment implements LLamaAPI.ModelStateListener {
    
    private static final String TAG = "AiChatFragment";
    
    private RecyclerView messagesRecyclerView;
    private EditText messageInput;
    private ImageButton sendButton;
    private ImageButton clearButton;
    private MessageAdapter messageAdapter;
    private ChatDbHelper dbHelper;
    private LLamaAPI chatLlamaApi; // 专用于聊天的LLamaAPI实例
    private Handler mainHandler;
    private boolean isGenerating = false;
    private long lastUIUpdateTime = 0;
    private static final long UI_UPDATE_INTERVAL = 50; // 毫秒
    private ConversationCounter counterManager;
    private PsychologicalStatusService psychologicalStatusService;
    private com.example.project.utils.PsychologicalStatusManager psychologicalStatusManager;
    
    // 添加静态变量，用于跨Fragment共享模型实例
    private static LLamaAPI sharedChatLlamaApi;
    


    public static AiChatFragment newInstance() {
        return new AiChatFragment();
    }
    
    /**
     * 显示模型加载对话框
     */
    private void showModelLoadingDialog() {
        if (!isAdded() || getContext() == null) {
            return;
        }
        
        ModelLoadingDialogManager.getInstance().showModelLoadingDialog(
            getContext(),
            () -> loadSelectedModel("default") // 立即加载（强制）
        );
    }
    
    /**
     * 加载选定的模型
     */
    private void loadSelectedModel(String modelType) {
        if (!isAdded() || getContext() == null) {
            return;
        }
        
        // 确保mainHandler已初始化
        if (mainHandler == null) {
            mainHandler = new Handler(Looper.getMainLooper());
        }
        
        // 创建加载进度对话框
        AlertDialog.Builder builder = new AlertDialog.Builder(requireContext());
        builder.setTitle("正在加载模型");
        builder.setMessage("正在检查模型文件...");
        builder.setCancelable(false);
        
        AlertDialog progressDialog = builder.create();
        progressDialog.show();
        
        // 在后台线程中加载模型
        new Thread(() -> {
            try {
                // 构建模型文件路径
                File filesDir = getContext().getFilesDir();
                File modelsDir = new File(filesDir, "models");
                String chatModelPath = new File(modelsDir, "Bluecat_chat.gguf").getAbsolutePath();
                String statusModelPath = new File(modelsDir, "Bluecat_status.gguf").getAbsolutePath();
                
                // 检查聊天模型文件是否存在
                File chatModelFile = new File(chatModelPath);
                File statusModelFile = new File(statusModelPath);
                
                // 如果聊天模型文件不存在或无效，从assets复制
                if (!chatModelFile.exists() || chatModelFile.length() < 10 * 1024 * 1024 || !chatModelFile.canRead()) {
                    mainHandler.post(() -> {
                        if (progressDialog.isShowing()) {
                            progressDialog.setMessage("正在复制聊天模型文件...");
                        }
                    });
                    
                    // 确保目录存在
                    if (!modelsDir.exists()) {
                        modelsDir.mkdirs();
                    }
                    
                    // 从assets复制聊天模型文件
                    copyModelFromAssets("Bluecat_chat.gguf", chatModelPath);
                }
                
                // 如果状态判断模型文件不存在或无效，从assets复制
                if (!statusModelFile.exists() || statusModelFile.length() < 10 * 1024 * 1024 || !statusModelFile.canRead()) {
                    mainHandler.post(() -> {
                        if (progressDialog.isShowing()) {
                            progressDialog.setMessage("正在复制状态判断模型文件...");
                        }
                    });
                    
                    // 从assets复制状态判断模型文件
                    copyModelFromAssets("Bluecat_status.gguf", statusModelPath);
                }
                
                // 更新进度信息
                mainHandler.post(() -> {
                    if (progressDialog.isShowing()) {
                        progressDialog.setMessage("正在加载AI模型到内存...");
                    }
                });
                
                // 添加模型状态监听器
                chatLlamaApi.addModelStateListener(new LLamaAPI.ModelStateListener() {
                    @Override
                    public void onModelLoaded() {
                        mainHandler.post(() -> {
                            if (isAdded() && getContext() != null) {
                                if (progressDialog.isShowing()) {
                                    progressDialog.dismiss();
                                }
                                Toast.makeText(getContext(), "AI模型加载成功！可以开始对话了", Toast.LENGTH_LONG).show();
                                
                                // 启用发送按钮
                                if (sendButton != null) {
                                    sendButton.setEnabled(true);
                                }
                                

                            }
                        });
                    }
                    
                    @Override
                    public void onModelUnloaded() {
                        // 模型卸载时的处理
                    }
                });
                
                // 加载模型
                chatLlamaApi.loadModel(chatModelPath);
                
            } catch (Exception e) {
                Log.e(TAG, "加载模型时发生异常", e);
                mainHandler.post(() -> {
                    if (isAdded() && getContext() != null) {
                        progressDialog.dismiss();
                        Toast.makeText(getContext(), "模型加载失败: " + e.getMessage(), Toast.LENGTH_LONG).show();
                    }
                });
            }
        }).start();
    }
    
    // 从ProfileFragment获取共享的模型实例
    private LLamaAPI getSharedModelInstance() {
        try {
            // 尝试获取ProfileFragment中的共享实例
            Class<?> profileFragmentClass = Class.forName("com.example.project.fragment.ProfileFragment");
            java.lang.reflect.Field field = profileFragmentClass.getDeclaredField("sharedProfileLlamaApi");
            field.setAccessible(true);
            Object instance = field.get(null);
            if (instance instanceof LLamaAPI) {
                Log.d(TAG, "成功获取ProfileFragment中的共享模型实例");
                return (LLamaAPI) instance;
            }
        } catch (Exception e) {
            Log.e(TAG, "获取ProfileFragment共享模型实例失败", e);
        }
        return null;
    }

    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container,
                           Bundle savedInstanceState) {
        return inflater.inflate(R.layout.fragment_ai_chat, container, false);
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);
        
        // 初始化Handler
        mainHandler = new Handler(Looper.getMainLooper());
        
        // 优先使用ProfileFragment中的共享模型实例
        LLamaAPI profileSharedInstance = getSharedModelInstance();
        
        if (profileSharedInstance != null && profileSharedInstance.isModelLoaded()) {
            // 使用ProfileFragment中已加载的模型实例
            sharedChatLlamaApi = profileSharedInstance;
            Log.d(TAG, "使用ProfileFragment中已加载的模型实例");
        } else if (sharedChatLlamaApi == null) {
            // 如果没有可用的共享实例，创建新实例
            sharedChatLlamaApi = LLamaAPI.createInstance("chat_model");
            Log.d(TAG, "创建聊天专用LLamaAPI实例");
        }
        chatLlamaApi = sharedChatLlamaApi;
        
        // 注册监听器
        chatLlamaApi.addModelStateListener(this);
        
        // 设置更优的生成温度
        chatLlamaApi.setTemperature(0.7f);
        
        // 只有在首次创建时重置聊天会话，而不是每次进入页面
        if (savedInstanceState == null) {
            chatLlamaApi.resetChatSession();
        }
        
        // 检查模型状态并记录
        boolean modelLoaded = chatLlamaApi.isModelLoaded();
        Log.d(TAG, "聊天模型初始加载状态: " + modelLoaded);
        
        // 如果模型未加载，尝试自动加载
        if (!modelLoaded) {
            checkAndLoadModel();
        }
        
        // 初始化数据库
        dbHelper = new ChatDbHelper(requireContext());
        
        // 初始化对话计数器和心理状态评估服务
        counterManager = new ConversationCounter(requireContext());
        psychologicalStatusService = new PsychologicalStatusService(requireContext());
        psychologicalStatusManager = new com.example.project.utils.PsychologicalStatusManager(requireContext());
        
        // 初始化视图
        messagesRecyclerView = view.findViewById(R.id.messagesRecyclerView);
        messageInput = view.findViewById(R.id.messageInput);
        sendButton = view.findViewById(R.id.sendButton);
        clearButton = view.findViewById(R.id.clearButton);

        // 设置RecyclerView
        messagesRecyclerView.setLayoutManager(new LinearLayoutManager(requireContext()));
        List<Message> messages = dbHelper.getAllMessages();
        messageAdapter = new MessageAdapter(messages);
        messagesRecyclerView.setAdapter(messageAdapter);

        // 设置发送按钮点击事件
        sendButton.setOnClickListener(v -> sendMessage());
        
        // 设置清空按钮点击事件
        clearButton.setOnClickListener(v -> clearChatHistory());
        
        // 添加长按发送按钮清除历史记录的功能
        sendButton.setOnLongClickListener(v -> {
            clearChatHistory();
            return true;
        });
    }

    private void sendMessage() {
        String content = messageInput.getText().toString().trim();
        if (!content.isEmpty()) {
            // 在发送消息前再次尝试获取共享模型实例
            if (!chatLlamaApi.isModelLoaded()) {
                // 尝试获取ProfileFragment中的共享实例
                LLamaAPI profileSharedInstance = getSharedModelInstance();
                if (profileSharedInstance != null && profileSharedInstance.isModelLoaded()) {
                    // 更新为已加载的模型实例
                    sharedChatLlamaApi = profileSharedInstance;
                    chatLlamaApi = sharedChatLlamaApi;
                    Log.d(TAG, "发送消息前更新为ProfileFragment中已加载的模型实例");
                }
            }
            
            // 再次检查模型是否已加载
            boolean modelLoaded = chatLlamaApi.isModelLoaded();
            Log.d(TAG, "聊天前检查模型: isModelLoaded = " + modelLoaded);
            
            if (!modelLoaded) {
                if (isAdded() && getContext() != null) {
                    showModelLoadingDialog();
                }
                return;
            }
            
            // 避免重复生成
            if (isGenerating) {
                if (isAdded() && getContext() != null) {
                    Toast.makeText(getContext(), "AI正在思考中，请稍候...", Toast.LENGTH_SHORT).show();
                }
                return;
            }
            
            // 保存并显示用户消息到数据库
            Message userMessage = new Message(content, false);
            dbHelper.insertMessage(userMessage);
            messageAdapter.addMessage(userMessage);

            // 增加对话计数
            counterManager.incrementConversationCount();
            Log.d(TAG, "对话计数: " + counterManager.getCurrentCount() + 
                  ", 距离下次分析还需: " + counterManager.getRemainingCountForNextAnalysis() + "轮对话");
            
            // 清空输入框
            messageInput.setText("");

            // 滚动到底部
            messagesRecyclerView.smoothScrollToPosition(messageAdapter.getItemCount() - 1);

            // 显示AI正在输入的状态，同时插入到数据库
            Message aiMessage = new Message("AI思考中...", true);
            // 先插入到数据库获取ID
            dbHelper.insertMessage(aiMessage);
            messageAdapter.addMessage(aiMessage);
            messagesRecyclerView.smoothScrollToPosition(messageAdapter.getItemCount() - 1);
            
            // 禁用发送按钮
            isGenerating = true;
            sendButton.setEnabled(false);
            
            // 使用LLamaAPI生成回复
            StringBuilder responseBuilder = new StringBuilder();
            long startTime = System.currentTimeMillis();
            final AtomicInteger tokenCount = new AtomicInteger(0);

            chatLlamaApi.chat(content, new LLamaAPI.CompletionCallback() {
                @Override
                public void onToken(String token) {
                    if (token != null && !token.isEmpty()) {
                        responseBuilder.append(token);
                        tokenCount.incrementAndGet();
                        
                        // 使用时间间隔控制UI更新频率
                        long currentTime = System.currentTimeMillis();
                        if (currentTime - lastUIUpdateTime > UI_UPDATE_INTERVAL) {
                            mainHandler.post(() -> {
                                if (isAdded()) {
                                    aiMessage.setContent(responseBuilder.toString());
                                    messageAdapter.notifyItemChanged(messageAdapter.getItemCount() - 1);
                                    
                                    // 只在需要时滚动
                                    LinearLayoutManager layoutManager = 
                                        (LinearLayoutManager) messagesRecyclerView.getLayoutManager();
                                    int position = layoutManager.findLastVisibleItemPosition();
                                    int count = messageAdapter.getItemCount();
                                    if (position >= count - 3) {
                                        messagesRecyclerView.smoothScrollToPosition(messageAdapter.getItemCount() - 1);
                                    }
                                }
                            });
                            lastUIUpdateTime = currentTime;
                        }
                    }
                }

                @Override
                public void onComplete() {
                    long endTime = System.currentTimeMillis();
                    float seconds = (endTime - startTime) / 1000f;
                    float tokensPerSecond = tokenCount.get() / Math.max(seconds, 0.1f);
                    
                    // 立即清空所有待处理的UI更新
                    mainHandler.removeCallbacksAndMessages(null);
                    
                    mainHandler.post(() -> {
                        if (isAdded()) {
                            // 更新最终结果并保存到数据库
                            String finalContent = responseBuilder.toString();
                            aiMessage.setContent(finalContent);
                            dbHelper.updateMessage(aiMessage);
                            messageAdapter.notifyItemChanged(messageAdapter.getItemCount() - 1);
                            
                            // 日志记录性能
                            Log.d(TAG, String.format("生成完成，内容长度: %d字符, %d个token, 用时: %.1f秒, 速度: %.1f tokens/秒", 
                                  finalContent.length(), tokenCount.get(), seconds, tokensPerSecond));
                            
                            // 增加对话计数
                            int currentCount = counterManager.incrementConversationCount();
                            Log.d(TAG, "对话计数增加到: " + currentCount);
                            
                            // 更新对话状态指示器
                            showChatHistoryStatus();
                            
                            // 重新启用发送按钮
                            isGenerating = false;
                            sendButton.setEnabled(true);
                            
                            // 检查是否需要进行心理状态评估
                            if (counterManager.shouldPerformAnalysis()) {
                                Log.d(TAG, "触发心理状态评估，当前对话计数: " + currentCount);
                                // 延迟1500毫秒执行心理状态评估，确保AI消息回复完成且气泡显示完全
                                mainHandler.postDelayed(() -> {
                                    if (isAdded()) {
                                        Log.d(TAG, "延迟执行心理状态评估，确保消息气泡显示完全");
                                        performPsychologicalAnalysis();
                                    }
                                }, 1500);
                            }
                        }
                    });
                }

                @Override
                public void onError(Exception e) {
                    mainHandler.post(() -> {
                        if (isAdded() && getContext() != null) {
                            Log.e(TAG, "Chat error", e);
                            String errorMessage = "生成失败: " + e.getMessage();
                            Toast.makeText(getContext(), errorMessage, Toast.LENGTH_SHORT).show();
                            
                            // 更新消息内容和数据库
                            aiMessage.setContent(errorMessage);
                            dbHelper.updateMessage(aiMessage);
                            messageAdapter.notifyItemChanged(messageAdapter.getItemCount() - 1);
                            
                            // 重新启用发送按钮
                            isGenerating = false;
                            sendButton.setEnabled(true);
                        }
                    });
                }
            });
        }
    }

    @Override
    public void onResume() {
        super.onResume();
        
        // 在页面恢复时尝试获取最新的共享模型实例
        LLamaAPI profileSharedInstance = getSharedModelInstance();
        if (profileSharedInstance != null && profileSharedInstance.isModelLoaded()) {
            // 如果ProfileFragment中有已加载的模型实例，使用该实例
            if (sharedChatLlamaApi != profileSharedInstance) {
                // 先移除旧实例的监听器
                if (chatLlamaApi != null) {
                    chatLlamaApi.removeModelStateListener(this);
                }
                
                // 更新为新的共享实例
                sharedChatLlamaApi = profileSharedInstance;
                chatLlamaApi = sharedChatLlamaApi;
                
                // 为新实例添加监听器
                chatLlamaApi.addModelStateListener(this);
                
                Log.d(TAG, "onResume: 更新为ProfileFragment中已加载的模型实例");
            }
        }
        
        // 检查并记录当前模型状态
        boolean modelLoaded = chatLlamaApi != null && chatLlamaApi.isModelLoaded();
        Log.d(TAG, "onResume: 当前模型加载状态 = " + modelLoaded);
        
        // 更新UI状态
        if (sendButton != null) {
            sendButton.setEnabled(modelLoaded);
        }
    }
    
    @Override
    public void onDestroyView() {
        super.onDestroyView();
        // 清理资源
        if (dbHelper != null) {
            dbHelper.close();
        }
        // 移除监听器，但不卸载模型
        if (chatLlamaApi != null) {
            chatLlamaApi.removeModelStateListener(this);
        }
        
        // 释放心理状态评估服务资源，但不卸载模型
        if (psychologicalStatusService != null) {
            // 不再调用release方法，避免卸载模型
            // psychologicalStatusService.release();
            Log.d(TAG, "保留心理状态评估服务资源，避免重新加载模型");
        }
    }
    
    // 实现ModelStateListener接口
    @Override
    public void onModelLoaded() {
        // 模型已加载，可以更新UI状态
        Log.d(TAG, "onModelLoaded callback received");
        mainHandler.post(() -> {
            if (isAdded()) {
                // 可以添加视觉提示表明模型已加载
                sendButton.setEnabled(true);
                
                if (getContext() != null) {
                    String modelName = chatLlamaApi != null ? chatLlamaApi.getCurrentModelName() : null;
                    String modelMessage;
                    
                    if (modelName != null) {
                        if (modelName.contains("QwQ")) {
                            modelMessage = "小模型 (QwQ-0.5B) 已加载完成，可以开始对话";
                        } else if (modelName.contains("Minicpm")) {
                            modelMessage = "大模型 (Minicpm-4B) 已加载完成，可以开始对话";
                        } else {
                            modelMessage = "模型 " + modelName + " 已加载完成，可以开始对话";
                        }
                    } else {
                        modelMessage = "AI模型已加载完成，可以开始对话";
                    }
                    
                    // 显示模型加载完成的提示
                    Toast.makeText(getContext(), modelMessage, Toast.LENGTH_SHORT).show();
                }
                

            }
        });
    }
    
    @Override
    public void onModelUnloaded() {
        // 模型已卸载，可以更新UI状态
        Log.d(TAG, "onModelUnloaded callback received");
        mainHandler.post(() -> {
            if (isAdded() && getContext() != null) {
                // 检查是否是用户主动卸载模型，只有在这种情况下才显示Toast
                // 页面跳转导致的监听器触发不应显示Toast
                if (sharedChatLlamaApi == null) {
                    Toast.makeText(getContext(), "模型已卸载，需要重新加载才能使用AI对话", Toast.LENGTH_SHORT).show();
                } else {
                    Log.d(TAG, "模型实例仍然存在，不显示卸载提示");
                }
            }
        });
    }

    // 添加一个重置聊天历史的方法
    private void clearChatHistory() {
        new AlertDialog.Builder(requireContext())
            .setTitle("清除聊天历史")
            .setMessage("是否要清除所有聊天历史？AI将不再记得之前的对话内容。")
            .setPositiveButton("确定", (dialog, which) -> {
                // 清除LLamaAPI内部历史记录
                chatLlamaApi.resetChatSession(true);
                
                // 清除数据库中的所有消息
                int deletedCount = dbHelper.deleteAllMessages();
                Log.d(TAG, "已从数据库中删除 " + deletedCount + " 条消息");
                
                // 重置对话计数器
                counterManager.resetCount();
                Log.d(TAG, "对话计数已重置");
                
                // 更新UI
                if (messageAdapter != null) {
                    messageAdapter.clearMessages();
                    messageAdapter.notifyDataSetChanged();
                }
                
                Toast.makeText(requireContext(), "聊天历史已清除", Toast.LENGTH_SHORT).show();
            })
            .setNegativeButton("取消", null)
            .show();
    }

    // 显示当前对话状态
    private void showChatHistoryStatus() {
        if (isAdded() && chatLlamaApi != null) {
            int historySize = chatLlamaApi.getChatHistorySize();
            
            if (historySize > 2) {
                // 计算轮数（一轮是用户+AI的对话）
                int rounds = historySize / 2;
                String status = "AI已记忆" + rounds + "轮对话";
                
                // 只记录到日志，不打扰用户
                Log.d(TAG, status + ", 历史记录长度: " + historySize);
            }
        }
    }
    
    /**
     * 执行心理状态评估
     */
    private void performPsychologicalAnalysis() {
        // 显示加载对话框
        if (isAdded() && getContext() != null && psychologicalStatusService != null) {
            // 使用AlertDialog替代已弃用的ProgressDialog
            AlertDialog.Builder builder = new AlertDialog.Builder(getContext());
            builder.setMessage("正在进行心理状态评估...");
            builder.setCancelable(false);
            final AlertDialog dialog = builder.create();
            dialog.show();
            
            try {
                psychologicalStatusService.analyzeUserPsychologicalStatus(new PsychologicalStatusService.AnalysisCallback() {
                    @Override
                    public void onSuccess(String analysisResult) {
                        if (isAdded() && getContext() != null) {
                            // 关闭加载对话框
                            dialog.dismiss();
                            Log.d(TAG, "心理状态评估结果: " + analysisResult);
                            
                            // 保存评估结果到本地
                            boolean saved = psychologicalStatusManager.saveStatusResult(analysisResult);
                            if (saved) {
                                // 在日志中显示完整的评估结果历史JSON字符串
                                String historyJson = psychologicalStatusManager.getStatusHistory();
                                Log.d(TAG, "心理状态评估历史记录: " + historyJson);
                            } else {
                                Log.e(TAG, "保存心理状态评估结果失败");
                            }
                            
                            // 显示评估结果给用户
                            showPsychologicalAnalysisResult(analysisResult);
                        }
                    }
                    
                    @Override
                    public void onError(Exception e) {
                        if (isAdded() && getContext() != null) {
                            // 关闭加载对话框
                            dialog.dismiss();
                            Log.e(TAG, "心理状态评估失败", e);
                            Toast.makeText(getContext(), "心理状态评估失败: " + e.getMessage(), Toast.LENGTH_LONG).show();
                        }
                    }
                });
            } catch (Exception e) {
                // 关闭加载对话框
                dialog.dismiss();
                // 显示错误信息
                Log.e(TAG, "心理状态评估失败", e);
                Toast.makeText(getContext(), "心理状态评估失败: " + e.getMessage(), Toast.LENGTH_LONG).show();
            }
        }
    }
    
    // 显示心理状态评估结果
    private void showPsychologicalAnalysisResult(String result) {
        Log.d(TAG, "心理状态评估结果: " + result);
        
        // 默认值
        int depressionLevel = 0; // 默认为正常
        int anxietyLevel = 0;    // 默认为正常
        String depressionState = "正常";
        String anxietyState = "正常";
        boolean isJsonFormat = false;
        
        try {
            // 尝试解析JSON格式 {"depression": X, "anxiety": Y}
            if (result != null && result.contains("depression") && result.contains("anxiety")) {
                Log.d(TAG, "尝试解析JSON格式的评估结果");
                // 提取JSON部分
                int startIndex = result.indexOf('{');
                int endIndex = result.indexOf('}', startIndex) + 1;
                
                if (startIndex >= 0 && endIndex > startIndex) {
                    String jsonStr = result.substring(startIndex, endIndex);
                    Log.d(TAG, "提取的JSON字符串: " + jsonStr);
                    
                    // 使用正则表达式提取depression和anxiety的值
                    java.util.regex.Pattern depressionPattern = java.util.regex.Pattern.compile("\"depression\"\\s*:\\s*(\\d)");
                    java.util.regex.Pattern anxietyPattern = java.util.regex.Pattern.compile("\"anxiety\"\\s*:\\s*(\\d)");
                    
                    java.util.regex.Matcher depressionMatcher = depressionPattern.matcher(jsonStr);
                    java.util.regex.Matcher anxietyMatcher = anxietyPattern.matcher(jsonStr);
                    
                    if (depressionMatcher.find()) {
                        depressionLevel = Integer.parseInt(depressionMatcher.group(1));
                        Log.d(TAG, "解析到抑郁级别: " + depressionLevel);
                    } else {
                        Log.d(TAG, "未找到抑郁级别");
                    }
                    
                    if (anxietyMatcher.find()) {
                        anxietyLevel = Integer.parseInt(anxietyMatcher.group(1));
                        Log.d(TAG, "解析到焦虑级别: " + anxietyLevel);
                    } else {
                        Log.d(TAG, "未找到焦虑级别");
                    }
                    
                    isJsonFormat = true;
                    
                    // 根据级别设置状态描述
                    switch (depressionLevel) {
                        case 0: depressionState = "正常"; break;
                        case 1: depressionState = "轻度"; break;
                        case 2: depressionState = "中度"; break;
                        case 3: depressionState = "重度"; break;
                    }
                    
                    switch (anxietyLevel) {
                        case 0: anxietyState = "正常"; break;
                        case 1: anxietyState = "轻度"; break;
                        case 2: anxietyState = "中度"; break;
                        case 3: anxietyState = "重度"; break;
                    }
                }
            } else {
                // 如果不是JSON格式，使用旧的文本解析方式作为备选
                Log.d(TAG, "未找到JSON格式，使用文本解析方式");
                if (result.contains("情绪状态：正常") || result.contains("情绪状态:正常") || 
                    result.contains("情绪状态为正常") || result.contains("情绪状态是正常")) {
                    depressionState = "正常";
                    anxietyState = "正常";
                    Log.d(TAG, "文本解析结果: 情绪状态正常");
                } else if (result.contains("情绪状态：一般") || result.contains("情绪状态:一般") || 
                           result.contains("情绪状态为一般") || result.contains("情绪状态是一般")) {
                    depressionState = "轻度";
                    anxietyState = "轻度";
                    Log.d(TAG, "文本解析结果: 情绪状态一般");
                } else if (result.contains("情绪状态：较差") || result.contains("情绪状态:较差") || 
                           result.contains("情绪状态为较差") || result.contains("情绪状态是较差")) {
                    depressionState = "中度";
                    anxietyState = "中度";
                    Log.d(TAG, "文本解析结果: 情绪状态较差");
                } else {
                    Log.d(TAG, "文本解析未匹配到已知情绪状态模式");
                }
            }
        } catch (Exception e) {
            Log.e(TAG, "解析心理状态评估结果失败", e);
        }
        
        Log.d(TAG, "最终评估结果 - 抑郁状态: " + depressionState + ", 焦虑状态: " + anxietyState + ", 是否JSON格式: " + isJsonFormat);
        
        // 使用final变量以便在lambda表达式中使用
        final String finalDepressionState = depressionState;
        final String finalAnxietyState = anxietyState;
        
        // 使用requireActivity().runOnUiThread确保UI操作在主线程执行
        if (isAdded() && getContext() != null) {
            requireActivity().runOnUiThread(() -> {
                // 显示抑郁和焦虑两个维度的评估结果
                AlertDialog dialog = new AlertDialog.Builder(requireContext())
                    .setTitle("心理状态评估")
                    .setMessage("抑郁状态：" + finalDepressionState + "\n焦虑状态：" + finalAnxietyState)
                    .setPositiveButton("了解", null)
                    .create();
                dialog.show();
                Log.d(TAG, "显示心理状态评估对话框");
            });
        }
    }
    
    /**
     * 检查模型状态并在需要时触发模型加载
     */
    private void checkAndLoadModel() {
        try {
            // 首先尝试获取ProfileFragment中的共享模型实例
            LLamaAPI profileSharedInstance = getSharedModelInstance();
            
            if (profileSharedInstance != null && profileSharedInstance.isModelLoaded()) {
                // 如果ProfileFragment中的模型已加载，直接使用
                sharedChatLlamaApi = profileSharedInstance;
                chatLlamaApi = sharedChatLlamaApi;
                Log.d(TAG, "使用ProfileFragment中已加载的模型实例");
                
                if (isAdded() && getContext() != null) {
                    Toast.makeText(getContext(), "模型已准备就绪，可以开始对话", Toast.LENGTH_SHORT).show();
                }
                return;
            }
            
            // 如果模型未加载，显示加载弹窗
            if (isAdded() && getContext() != null) {
                showModelLoadingDialog();
            }
        } catch (Exception e) {
            Log.e(TAG, "检查模型状态失败", e);
            if (isAdded() && getContext() != null) {
                Toast.makeText(getContext(), "检查模型状态失败: " + e.getMessage(), Toast.LENGTH_SHORT).show();
            }
        }
    }
    
    /**
     * 从assets目录复制模型文件到应用内部存储
     */
    private void copyModelFromAssets(String assetFileName, String targetPath) throws IOException {
        InputStream inputStream = null;
        OutputStream outputStream = null;
        
        try {
            inputStream = getContext().getAssets().open("models/" + assetFileName);
            outputStream = new FileOutputStream(targetPath);
            
            byte[] buffer = new byte[8192];
            int bytesRead;
            long totalBytes = 0;
            
            while ((bytesRead = inputStream.read(buffer)) != -1) {
                outputStream.write(buffer, 0, bytesRead);
                totalBytes += bytesRead;
                
                // 每复制10MB更新一次进度信息
                if (totalBytes % (10 * 1024 * 1024) == 0) {
                    final long currentBytes = totalBytes;
                    mainHandler.post(() -> {
                        if (isAdded() && getContext() != null) {
                            // 这里可以更新进度信息，但为了简化就不显示具体进度了
                        }
                    });
                }
            }
            
            Log.d(TAG, "成功复制模型文件: " + assetFileName + " (" + totalBytes + " bytes)");
            
        } finally {
            if (inputStream != null) {
                try {
                    inputStream.close();
                } catch (IOException e) {
                    Log.e(TAG, "关闭输入流失败", e);
                }
            }
            if (outputStream != null) {
                try {
                    outputStream.close();
                } catch (IOException e) {
                    Log.e(TAG, "关闭输出流失败", e);
                }
            }
        }
    }
}