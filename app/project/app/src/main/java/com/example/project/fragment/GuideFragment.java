package com.example.project.fragment;

import android.app.AlertDialog;
import android.content.Context;
import android.content.SharedPreferences;
import android.os.Bundle;
import android.text.Editable;
import android.text.TextWatcher;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import com.example.project.R;
import com.example.project.adapter.GuideAdapter;
import com.example.project.db.GuideDbHelper;
import com.example.project.model.Guide;
import com.google.android.material.floatingactionbutton.FloatingActionButton;

import java.util.List;

public class GuideFragment extends Fragment implements GuideAdapter.OnGuideCompletionListener {
    
    private RecyclerView rvGuides;
    private FloatingActionButton fabAddGuide;
    private GuideAdapter guideAdapter;
    private GuideDbHelper guideDbHelper;
    private long userId;
    
    public static GuideFragment newInstance() {
        return new GuideFragment();
    }

    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container,
                           Bundle savedInstanceState) {
        return inflater.inflate(R.layout.fragment_guide, container, false);
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);
        
        // 获取当前用户ID
        SharedPreferences prefs = requireActivity().getSharedPreferences("user_info", Context.MODE_PRIVATE);
        userId = prefs.getLong("user_id", 0);
        
        // 初始化数据库帮助类
        guideDbHelper = new GuideDbHelper(requireContext());
        
        // 初始化视图
        initViews(view);
        
        // 加载指南数据
        loadGuides();
    }
    
    private void initViews(View view) {
        rvGuides = view.findViewById(R.id.rv_guides);
        fabAddGuide = view.findViewById(R.id.fab_add_guide);
        
        // 设置RecyclerView
        rvGuides.setLayoutManager(new LinearLayoutManager(requireContext()));
        
        // 设置添加按钮点击事件
        fabAddGuide.setOnClickListener(v -> showAddGuideDialog());
    }
    
    private void loadGuides() {
        // 从数据库加载指南数据
        List<Guide> guides = guideDbHelper.getGuidesForUser(userId);
        
        // 设置适配器
        if (guideAdapter == null) {
            guideAdapter = new GuideAdapter(guides, this);
            rvGuides.setAdapter(guideAdapter);
        } else {
            guideAdapter.updateGuides(guides);
        }
    }
    
    private void showAddGuideDialog() {
        // 创建对话框
        AlertDialog.Builder builder = new AlertDialog.Builder(requireContext());
        View dialogView = getLayoutInflater().inflate(R.layout.dialog_add_guide, null);
        builder.setView(dialogView);
        
        // 获取对话框视图
        EditText etGuideContent = dialogView.findViewById(R.id.et_guide_content);
        TextView tvCharCount = dialogView.findViewById(R.id.tv_char_count);
        
        // 设置字符计数器
        etGuideContent.addTextChangedListener(new TextWatcher() {
            @Override
            public void beforeTextChanged(CharSequence s, int start, int count, int after) {}
            
            @Override
            public void onTextChanged(CharSequence s, int start, int before, int count) {}
            
            @Override
            public void afterTextChanged(Editable s) {
                tvCharCount.setText(s.length() + "/100");
            }
        });
        
        // 设置对话框按钮
        builder.setPositiveButton("添加", (dialog, which) -> {
            String content = etGuideContent.getText().toString().trim();
            if (content.isEmpty()) {
                Toast.makeText(requireContext(), "治疗建议内容不能为空", Toast.LENGTH_SHORT).show();
                return;
            }
            
            // 创建新指南并保存到数据库
            Guide guide = new Guide(content, false, userId);
            guideDbHelper.insertGuide(guide);
            
            // 重新加载指南列表
            loadGuides();
            
            Toast.makeText(requireContext(), "添加成功", Toast.LENGTH_SHORT).show();
        });
        
        builder.setNegativeButton("取消", null);
        
        // 显示对话框
        AlertDialog dialog = builder.create();
        dialog.show();
    }
    
    @Override
    public void onGuideCompletionChanged(Guide guide, boolean isCompleted) {
        // 更新指南完成状态
        guideDbHelper.updateGuideCompletionStatus(guide.getId(), isCompleted);
        guide.setCompleted(isCompleted);
    }
}