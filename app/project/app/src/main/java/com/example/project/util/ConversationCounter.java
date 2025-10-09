package com.example.project.util;

import android.content.Context;
import android.util.Log;

/**
 * 对话计数器
 * 用于跟踪对话轮次并在达到指定次数时触发心理状态分析
 */
public class ConversationCounter {
    private static final String TAG = "ConversationCounter";
    
    // 每进行多少轮对话触发一次心理状态分析
    private static final int ANALYSIS_INTERVAL = 5;
    
    // 当前对话计数
    private int currentCount = 0;
    
    // 上下文对象
    private Context context;
    
    /**
     * 构造函数
     */
    public ConversationCounter() {
        Log.d(TAG, "创建对话计数器");
    }
    
    /**
     * 带Context的构造函数
     * @param context 上下文对象
     */
    public ConversationCounter(Context context) {
        this.context = context;
        Log.d(TAG, "创建对话计数器(带Context)");
    }
    
    /**
     * 增加对话计数
     * @return 增加后的计数值
     */
    public int incrementConversationCount() {
        currentCount++;
        Log.d(TAG, "对话计数增加到: " + currentCount);
        return currentCount;
    }
    
    /**
     * 获取当前对话计数
     * @return 当前对话计数
     */
    public int getCurrentCount() {
        return currentCount;
    }
    
    /**
     * 重置对话计数
     */
    public void resetCount() {
        currentCount = 0;
        Log.d(TAG, "对话计数已重置");
    }
    
    /**
     * 重置对话计数（别名方法，与resetCount功能相同）
     */
    public void resetConversationCount() {
        resetCount();
    }
    
    /**
     * 检查是否应该执行心理状态分析
     * @return 如果当前对话计数是ANALYSIS_INTERVAL的倍数，返回true；否则返回false
     */
    public boolean shouldPerformAnalysis() {
        return currentCount > 0 && currentCount % ANALYSIS_INTERVAL == 0;
    }
    
    /**
     * 获取距离下一次分析还需要的对话轮次
     * @return 距离下一次分析还需要的对话轮次
     */
    public int getRemainingCountForNextAnalysis() {
        if (currentCount == 0) {
            return ANALYSIS_INTERVAL;
        }
        
        int remainder = currentCount % ANALYSIS_INTERVAL;
        if (remainder == 0) {
            return 0; // 当前就应该执行分析
        } else {
            return ANALYSIS_INTERVAL - remainder;
        }
    }
}