package com.example.project.fragment;

import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import androidx.swiperefreshlayout.widget.SwipeRefreshLayout;

import com.example.project.R;
import com.example.project.adapter.NewsAdapter;
import com.example.project.api.ApiClient;
import com.example.project.model.News;

import java.util.ArrayList;
import java.util.List;

import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;

public class NewsFragment extends Fragment {
    
    private SwipeRefreshLayout swipeRefreshLayout;
    private RecyclerView newsRecyclerView;
    private NewsAdapter newsAdapter;

    public static NewsFragment newInstance() {
        return new NewsFragment();
    }

    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container,
                           Bundle savedInstanceState) {
        return inflater.inflate(R.layout.fragment_news, container, false);
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);
        
        // 初始化视图
        swipeRefreshLayout = view.findViewById(R.id.swipeRefreshLayout);
        newsRecyclerView = view.findViewById(R.id.newsRecyclerView);

        // 设置RecyclerView
        newsRecyclerView.setLayoutManager(new LinearLayoutManager(requireContext()));
        newsAdapter = new NewsAdapter(new ArrayList<>());
        newsRecyclerView.setAdapter(newsAdapter);

        // 设置下拉刷新 - 禁用刷新功能
        swipeRefreshLayout.setEnabled(false);

        // 不再自动加载新闻数据
        // loadNews();
    }

    // 保留但不再调用的加载新闻方法
    private void loadNews() {
        // 不再调用API获取新闻
        if (isAdded()) {
            swipeRefreshLayout.setRefreshing(false);
        }
    }

    // 保留但不再使用的刷新方法
    private void refreshNews() {
        // 不再调用API刷新新闻
        if (isAdded()) {
            swipeRefreshLayout.setRefreshing(false);
        }
    }

    private void showError(String message) {
        if (getContext() != null && isAdded()) {
            Toast.makeText(getContext(), message, Toast.LENGTH_SHORT).show();
        }
    }
}