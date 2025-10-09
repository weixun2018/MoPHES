package com.example.project.utils;

import android.content.Context;
import android.content.SharedPreferences;
import android.util.Log;

/**
 * 对话计数器类，用于跟踪用户与AI的对话轮次
 * 每五轮对话后触发心理状态评估
 */
public class ConversationCounter {
    private static final String TAG = "ConversationCounter";
    private static final String PREF_NAME = "conversation_counter";
    private static final String KEY_COUNT = "current_count";
    private static final int ANALYSIS_INTERVAL = 5; // 每5轮对话进行一次心理状态评估
    
    private final SharedPreferences preferences;
    
    public ConversationCounter(Context context) {
        preferences = context.getSharedPreferences(PREF_NAME, Context.MODE_PRIVATE);
    }
    
    /**
     * 增加对话计数
     * @return 当前对话计数
     */
    public int incrementConversationCount() {
        int currentCount = getCurrentCount() + 1;
        preferences.edit().putInt(KEY_COUNT, currentCount).apply();
        Log.d(TAG, "对话计数增加到: " + currentCount);
        return currentCount;
    }
    
    /**
     * 获取当前对话计数
     * @return 当前对话计数
     */
    public int getCurrentCount() {
        return preferences.getInt(KEY_COUNT, 0);
    }
    
    /**
     * 重置对话计数
     */
    public void resetCount() {
        preferences.edit().putInt(KEY_COUNT, 0).apply();
        Log.d(TAG, "对话计数已重置");
    }
    
    /**
     * 检查是否需要进行心理状态评估
     * @return 是否需要评估
     */
    public boolean shouldPerformAnalysis() {
        int currentCount = getCurrentCount();
        return currentCount > 0 && currentCount % ANALYSIS_INTERVAL == 0;
    }
    
    /**
     * 获取距离下次分析还需要的对话轮次
     * @return 剩余轮次
     */
    public int getRemainingCountForNextAnalysis() {
        int currentCount = getCurrentCount();
        return ANALYSIS_INTERVAL - (currentCount % ANALYSIS_INTERVAL);
    }
}