package com.example.project;

import android.app.Application;
import com.example.project.api.ApiClient;

public class ProjectApplication extends Application {
    
    @Override
    public void onCreate() {
        super.onCreate();
        
        // 初始化ApiClient
        ApiClient.init(this);
    }
} 