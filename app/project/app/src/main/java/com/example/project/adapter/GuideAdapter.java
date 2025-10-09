package com.example.project.adapter;

import android.graphics.Color;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.CheckBox;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;

import com.example.project.R;
import com.example.project.model.Guide;

import java.util.List;

public class GuideAdapter extends RecyclerView.Adapter<GuideAdapter.GuideViewHolder> {

    private List<Guide> guideList;
    private OnGuideCompletionListener listener;

    public interface OnGuideCompletionListener {
        void onGuideCompletionChanged(Guide guide, boolean isCompleted);
    }

    public GuideAdapter(List<Guide> guideList, OnGuideCompletionListener listener) {
        this.guideList = guideList;
        this.listener = listener;
    }

    @NonNull
    @Override
    public GuideViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View view = LayoutInflater.from(parent.getContext()).inflate(R.layout.item_guide, parent, false);
        return new GuideViewHolder(view);
    }

    @Override
    public void onBindViewHolder(@NonNull GuideViewHolder holder, int position) {
        Guide guide = guideList.get(position);
        holder.bind(guide);
    }

    @Override
    public int getItemCount() {
        return guideList.size();
    }

    public void updateGuides(List<Guide> newGuideList) {
        this.guideList = newGuideList;
        notifyDataSetChanged();
    }

    class GuideViewHolder extends RecyclerView.ViewHolder {
        private TextView tvGuideContent;
        private CheckBox cbGuideCompleted;

        public GuideViewHolder(@NonNull View itemView) {
            super(itemView);
            tvGuideContent = itemView.findViewById(R.id.tv_guide_content);
            cbGuideCompleted = itemView.findViewById(R.id.cb_guide_completed);
        }

        public void bind(final Guide guide) {
            tvGuideContent.setText(guide.getContent());
            cbGuideCompleted.setChecked(guide.isCompleted());
            
            // 如果指南已完成，将文本颜色设置为浅灰色
            if (guide.isCompleted()) {
                tvGuideContent.setTextColor(Color.LTGRAY);
            } else {
                tvGuideContent.setTextColor(itemView.getContext().getResources().getColor(R.color.telegram_text_primary));
            }

            // 设置复选框的点击事件
            cbGuideCompleted.setOnClickListener(v -> {
                boolean isChecked = cbGuideCompleted.isChecked();
                if (listener != null) {
                    listener.onGuideCompletionChanged(guide, isChecked);
                }
                
                // 更新文本颜色
                if (isChecked) {
                    tvGuideContent.setTextColor(Color.LTGRAY);
                } else {
                    tvGuideContent.setTextColor(itemView.getContext().getResources().getColor(R.color.telegram_text_primary));
                }
            });
        }
    }
}