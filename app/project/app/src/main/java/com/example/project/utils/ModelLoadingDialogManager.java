package com.example.project.utils;

import android.app.AlertDialog;
import android.content.Context;
import android.util.Log;

/**
 * 全局单例对话框管理器，确保同时只有一个模型加载对话框
 * 强制用户必须加载AI模型，不提供"稍后"选项
 */
public class ModelLoadingDialogManager {
    private static final String TAG = "ModelLoadingDialogManager";
    private static ModelLoadingDialogManager instance;
    private AlertDialog currentDialog;
    
    private ModelLoadingDialogManager() {}
    
    public static synchronized ModelLoadingDialogManager getInstance() {
        if (instance == null) {
            instance = new ModelLoadingDialogManager();
        }
        return instance;
    }
    
    /**
     * 显示模型加载对话框（强制加载，无"稍后"选项）
     * @param context 上下文
     * @param onLoadNow 点击"立即加载"的回调
     * @return 是否成功显示对话框（如果已有对话框显示，则返回false）
     */
    public boolean showModelLoadingDialog(Context context, Runnable onLoadNow) {
        // 如果已有对话框显示，不重复显示
        if (currentDialog != null && currentDialog.isShowing()) {
            Log.d(TAG, "已有模型加载对话框显示，不重复显示");
            return false;
        }
        
        Log.d(TAG, "显示强制模型加载对话框");
        
        AlertDialog.Builder builder = new AlertDialog.Builder(context);
        builder.setTitle("加载AI模型");
        builder.setMessage("为了获得更好的聊天体验，需要加载AI模型。请点击下方按钮开始加载。");
        builder.setCancelable(false);
        
        builder.setPositiveButton("立即加载", (dialog, which) -> {
            Log.d(TAG, "用户选择立即加载模型");
            try {
                // 先显式关闭当前对话框，避免界面残留或引用竞争
                if (currentDialog != null && currentDialog.isShowing()) {
                    currentDialog.dismiss();
                }
            } catch (Exception e) {
                Log.w(TAG, "关闭对话框时发生异常", e);
            } finally {
                currentDialog = null;
            }

            if (onLoadNow != null) {
                // 确保回调在主线程执行，避免线程切换问题
                new android.os.Handler(android.os.Looper.getMainLooper()).post(onLoadNow);
            }
        });
        
        currentDialog = builder.create();
        currentDialog.show();
        return true;
    }
    

    
    /**
     * 关闭当前对话框
     */
    public void dismissCurrentDialog() {
        if (currentDialog != null && currentDialog.isShowing()) {
            Log.d(TAG, "关闭当前对话框");
            currentDialog.dismiss();
            currentDialog = null;
        }
    }
    
    /**
     * 检查是否有对话框正在显示
     */
    public boolean isDialogShowing() {
        return currentDialog != null && currentDialog.isShowing();
    }
}