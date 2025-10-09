package com.example.project.utils;

import android.content.Context;
import android.content.SharedPreferences;
import android.util.Log;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;

/**
 * 心理状态评估结果管理工具类
 * 负责存储和管理用户的心理状态评估结果
 */
public class PsychologicalStatusManager {
    private static final String TAG = "PsychologicalStatusManager";
    private static final String PREF_NAME = "psychological_status";
    private static final String KEY_STATUS_HISTORY = "status_history";
    
    private final Context context;
    private final SharedPreferences preferences;
    
    public PsychologicalStatusManager(Context context) {
        this.context = context;
        this.preferences = context.getSharedPreferences(PREF_NAME, Context.MODE_PRIVATE);
    }
    
    /**
     * 保存心理状态评估结果
     * @param result 评估结果JSON字符串
     * @return 是否保存成功
     */
    public boolean saveStatusResult(String result) {
        try {
            // 获取当前的评估结果历史记录
            JSONArray historyArray = getStatusHistoryArray();
            
            // 创建新的评估结果记录
            JSONObject statusRecord = new JSONObject();
            
            // 提取depression和anxiety值
            int depression = 0;
            int anxiety = 0;
            
            if (result != null && !result.trim().isEmpty()) {
                Log.d(TAG, "开始解析心理状态评估结果: " + result);
                
                // 首先尝试解析JSON格式
                boolean jsonParsed = false;
                try {
                    // 查找JSON格式的数据 {"depression": X, "anxiety": Y}
                    String jsonPattern = "\\{[^}]*\"depression\"[^}]*\"anxiety\"[^}]*\\}";
                    java.util.regex.Pattern pattern = java.util.regex.Pattern.compile(jsonPattern);
                    java.util.regex.Matcher matcher = pattern.matcher(result);
                    
                    if (matcher.find()) {
                        String jsonStr = matcher.group();
                        Log.d(TAG, "找到JSON格式数据: " + jsonStr);
                        
                        JSONObject jsonObj = new JSONObject(jsonStr);
                        depression = jsonObj.getInt("depression");
                        anxiety = jsonObj.getInt("anxiety");
                        jsonParsed = true;
                        Log.d(TAG, "JSON解析成功 - depression: " + depression + ", anxiety: " + anxiety);
                    }
                } catch (Exception e) {
                    Log.w(TAG, "JSON解析失败，尝试文本解析: " + e.getMessage());
                }
                
                // 如果JSON解析失败，使用文本解析
                if (!jsonParsed && result.contains("depression") && result.contains("anxiety")) {
                    Log.d(TAG, "使用文本解析方式");
                    String[] lines = result.split("\n");
                    for (String line : lines) {
                        line = line.trim();
                        if (line.contains("depression") && line.contains(":")) {
                            try {
                                String valueStr = line.substring(line.indexOf(":")+1).trim();
                                // 移除可能的逗号、空格和引号
                                valueStr = valueStr.replaceAll("[,\"\\s]", "");
                                depression = Integer.parseInt(valueStr);
                                Log.d(TAG, "文本解析depression成功: " + depression);
                            } catch (Exception e) {
                                Log.e(TAG, "解析depression值失败: " + line, e);
                            }
                        }
                        if (line.contains("anxiety") && line.contains(":")) {
                            try {
                                String valueStr = line.substring(line.indexOf(":")+1).trim();
                                // 移除可能的逗号、空格和引号
                                valueStr = valueStr.replaceAll("[,\"\\s]", "");
                                anxiety = Integer.parseInt(valueStr);
                                Log.d(TAG, "文本解析anxiety成功: " + anxiety);
                            } catch (Exception e) {
                                Log.e(TAG, "解析anxiety值失败: " + line, e);
                            }
                        }
                    }
                }
                
                Log.d(TAG, "最终解析结果 - depression: " + depression + ", anxiety: " + anxiety);
            }
            
            // 保存提取的值
            statusRecord.put("depression", depression);
            statusRecord.put("anxiety", anxiety);
            
            // 添加时间戳
            statusRecord.put("timestamp", System.currentTimeMillis());
            statusRecord.put("date", getCurrentDateString());
            
            // 不需要存储原始结果
            
            // 将新记录添加到历史记录中
            historyArray.put(statusRecord);
            
            // 保存更新后的历史记录
            SharedPreferences.Editor editor = preferences.edit();
            editor.putString(KEY_STATUS_HISTORY, historyArray.toString());
            boolean success = editor.commit();
            
            // 记录日志
            if (success) {
                Log.d(TAG, "心理状态评估结果已保存: " + statusRecord.toString());
                Log.d(TAG, "完整历史记录: " + historyArray.toString());
            } else {
                Log.e(TAG, "保存心理状态评估结果失败");
            }
            
            return success;
        } catch (JSONException e) {
            Log.e(TAG, "保存心理状态评估结果时出错", e);
            return false;
        }
    }
    
    /**
     * 获取所有心理状态评估结果历史
     * @return 评估结果历史的JSON字符串
     */
    public String getStatusHistory() {
        return preferences.getString(KEY_STATUS_HISTORY, "[]");
    }
    
    /**
     * 获取最新的心理状态评估结果（静态方法）
     * @param context 上下文
     * @return 最新的评估结果Map，包含depression、anxiety、timestamp等信息，如果没有记录则返回null
     */
    public static java.util.Map<String, Object> getLatestStatusResult(Context context) {
        try {
            SharedPreferences preferences = context.getSharedPreferences(PREF_NAME, Context.MODE_PRIVATE);
            String historyStr = preferences.getString(KEY_STATUS_HISTORY, "[]");
            JSONArray historyArray = new JSONArray(historyStr);
            
            if (historyArray.length() > 0) {
                // 获取最后一个（最新的）评估结果
                JSONObject latestResult = historyArray.getJSONObject(historyArray.length() - 1);
                
                // 转换为Map返回
                java.util.Map<String, Object> resultMap = new java.util.HashMap<>();
                resultMap.put("depression", latestResult.getInt("depression"));
                resultMap.put("anxiety", latestResult.getInt("anxiety"));
                resultMap.put("timestamp", latestResult.getString("date"));
                
                return resultMap;
            }
            
            return null;
        } catch (JSONException e) {
            Log.e(TAG, "获取最新心理状态评估结果失败", e);
            return null;
        }
    }
    
    /**
     * 获取心理状态评估结果历史JSONArray对象
     * @return 评估结果历史的JSONArray对象
     */
    private JSONArray getStatusHistoryArray() {
        try {
            String historyStr = preferences.getString(KEY_STATUS_HISTORY, "[]");
            return new JSONArray(historyStr);
        } catch (JSONException e) {
            Log.e(TAG, "解析心理状态评估历史记录失败", e);
            return new JSONArray();
        }
    }
    
    /**
     * 清除所有心理状态评估结果历史
     * @return 是否清除成功
     */
    public boolean clearStatusHistory() {
        SharedPreferences.Editor editor = preferences.edit();
        editor.remove(KEY_STATUS_HISTORY);
        boolean success = editor.commit();
        
        if (success) {
            Log.d(TAG, "心理状态评估历史记录已清除");
        } else {
            Log.e(TAG, "清除心理状态评估历史记录失败");
        }
        
        return success;
    }
    
    /**
     * 获取当前日期字符串
     * @return 格式化的日期字符串
     */
    private String getCurrentDateString() {
        SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault());
        return dateFormat.format(new Date());
    }
}