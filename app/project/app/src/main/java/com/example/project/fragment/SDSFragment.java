package com.example.project.fragment;

import android.content.Context;
import android.content.SharedPreferences;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.ProgressBar;
import android.widget.RadioButton;
import android.widget.RadioGroup;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;

import com.example.project.R;
import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;

import java.lang.reflect.Type;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class SDSFragment extends Fragment {
    
    private List<SDSQuestion> questions;
    private int currentQuestionIndex = 0;
    private Map<Integer, Integer> answers = new HashMap<>();
    private int userId = -1;
    private boolean isTestCompleted = false;
    
    // 视图组件
    private View questionLayout;
    private View resultLayout;
    private TextView questionProgress;
    private TextView questionText;
    private RadioGroup optionsGroup;
    private RadioButton option1, option2, option3, option4;
    private Button nextButton, previousButton, retestButton, clearAnswersButton;
    private ProgressBar progressBar;
    private TextView totalScoreText, resultInterpretationText, resultLevelText;
    
    // SharedPreferences 键名常量
    private static final String PREF_SDS_ANSWERS = "sds_answers";
    private static final String PREF_SDS_CURRENT_INDEX = "sds_current_index";
    private static final String PREF_SDS_IN_PROGRESS = "sds_in_progress";
    
    public static SDSFragment newInstance() {
        return new SDSFragment();
    }
    
    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container,
                             Bundle savedInstanceState) {
        View view = inflater.inflate(R.layout.fragment_sds, container, false);
        
        // 初始化视图
        initViews(view);
        
        // 设置问题布局可见，结果布局不可见
        questionLayout.setVisibility(View.GONE);
        resultLayout.setVisibility(View.GONE);
        progressBar.setVisibility(View.VISIBLE);
        
        return view;
    }
    
    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);
        
        // 获取用户ID - 使用getLong并转为int，避免类型转换异常
        SharedPreferences prefs = requireActivity().getSharedPreferences("user_info", Context.MODE_PRIVATE);
        userId = (int) prefs.getLong("user_id", -1);
        
        // 初始化问题列表
        initSDSQuestions();
        
        // 恢复之前的作答状态或显示第一题
        restoreAnswersFromLocal();
    }
    
    private void initViews(View view) {
        questionLayout = view.findViewById(R.id.questionLayout);
        resultLayout = view.findViewById(R.id.resultLayout);
        questionProgress = view.findViewById(R.id.questionProgress);
        questionText = view.findViewById(R.id.questionText);
        optionsGroup = view.findViewById(R.id.optionsGroup);
        option1 = view.findViewById(R.id.option1);
        option2 = view.findViewById(R.id.option2);
        option3 = view.findViewById(R.id.option3);
        option4 = view.findViewById(R.id.option4);
        nextButton = view.findViewById(R.id.nextButton);
        previousButton = view.findViewById(R.id.previousButton);
        retestButton = view.findViewById(R.id.retestButton);
        clearAnswersButton = view.findViewById(R.id.clearAnswersButton);
        progressBar = view.findViewById(R.id.progressBar);
        
        totalScoreText = view.findViewById(R.id.totalScoreText);
        resultInterpretationText = view.findViewById(R.id.resultInterpretationText);
        resultLevelText = view.findViewById(R.id.resultLevelText);
        
        nextButton.setOnClickListener(v -> handleNextQuestion());
        previousButton.setOnClickListener(v -> handlePreviousQuestion());
        retestButton.setOnClickListener(v -> restartTest());
        
        // 设置清空作答按钮点击事件
        if (clearAnswersButton != null) {
            clearAnswersButton.setOnClickListener(v -> clearAllAnswers());
        }
    }
    
    // 初始化SDS问题列表
    private void initSDSQuestions() {
        questions = new ArrayList<>();
        
        // 添加20个SDS问题
        questions.add(new SDSQuestion(1, "我感到情绪沮丧，郁闷"));
        questions.add(new SDSQuestion(2, "我感到早晨心情最好", true)); // 反向计分
        questions.add(new SDSQuestion(3, "我要哭或想哭"));
        questions.add(new SDSQuestion(4, "我夜间睡眠不好"));
        questions.add(new SDSQuestion(5, "我吃饭和平常一样多", true)); // 反向计分
        questions.add(new SDSQuestion(6, "我的性功能正常", true)); // 反向计分
        questions.add(new SDSQuestion(7, "我感到体重减轻"));
        questions.add(new SDSQuestion(8, "我为便秘烦恼"));
        questions.add(new SDSQuestion(9, "我的心跳比平时快"));
        questions.add(new SDSQuestion(10, "我无故感到疲劳"));
        questions.add(new SDSQuestion(11, "我的头脑和平常一样清楚", true)); // 反向计分
        questions.add(new SDSQuestion(12, "我做事情和平常一样不感到困难", true)); // 反向计分
        questions.add(new SDSQuestion(13, "我坐卧不安，难以保持平静"));
        questions.add(new SDSQuestion(14, "我对未来感到有希望", true)); // 反向计分
        questions.add(new SDSQuestion(15, "我比平常更容易激动"));
        questions.add(new SDSQuestion(16, "我觉得决定什么事很容易", true)); // 反向计分
        questions.add(new SDSQuestion(17, "我感到自己是有用的和不可缺少的人", true)); // 反向计分
        questions.add(new SDSQuestion(18, "我的生活很有意义", true)); // 反向计分
        questions.add(new SDSQuestion(19, "假若我死了别人会过得更好"));
        questions.add(new SDSQuestion(20, "我仍旧喜爱自己平常喜爱的东西", true)); // 反向计分
        
        // 显示第一题
        progressBar.setVisibility(View.GONE);
        questionLayout.setVisibility(View.VISIBLE);
        showQuestion(currentQuestionIndex);
    }
    
    // 显示指定索引的问题
    private void showQuestion(int index) {
        if (index < 0 || index >= questions.size()) {
            return;
        }
        
        currentQuestionIndex = index;
        SDSQuestion question = questions.get(index);
        
        // 更新问题进度
        questionProgress.setText(String.format("%d/%d", index + 1, questions.size()));
        
        // 更新问题文本
        questionText.setText(question.getContent());
        
        // 更新选项状态
        Integer selectedAnswer = answers.get(question.getId());
        optionsGroup.clearCheck();
        
        if (selectedAnswer != null) {
            switch (selectedAnswer) {
                case 1:
                    option1.setChecked(true);
                    break;
                case 2:
                    option2.setChecked(true);
                    break;
                case 3:
                    option3.setChecked(true);
                    break;
                case 4:
                    option4.setChecked(true);
                    break;
            }
        }
        
        // 更新按钮状态
        previousButton.setEnabled(index > 0);
        nextButton.setText(index == questions.size() - 1 ? "提交" : "下一题");
    }
    
    // 处理下一题按钮点击
    private void handleNextQuestion() {
        // 获取当前选中的选项
        int selectedOptionId = optionsGroup.getCheckedRadioButtonId();
        
        // 如果没有选择任何选项，提示用户
        if (selectedOptionId == -1) {
            Toast.makeText(getContext(), "请选择一个选项", Toast.LENGTH_SHORT).show();
            return;
        }
        
        // 保存当前答案
        int score;
        if (selectedOptionId == R.id.option1) {
            score = 1;
        } else if (selectedOptionId == R.id.option2) {
            score = 2;
        } else if (selectedOptionId == R.id.option3) {
            score = 3;
        } else {
            score = 4;
        }
        
        SDSQuestion currentQuestion = questions.get(currentQuestionIndex);
        
        // 如果是反向计分的题目，需要反转分数
        if (currentQuestion.isReversed()) {
            score = 5 - score; // 1->4, 2->3, 3->2, 4->1
        }
        
        answers.put(currentQuestion.getId(), score);
        
        // 如果是最后一题，计算结果
        if (currentQuestionIndex == questions.size() - 1) {
            calculateResult();
        } else {
            // 否则显示下一题
            showQuestion(currentQuestionIndex + 1);
        }
    }
    
    // 处理上一题按钮点击
    private void handlePreviousQuestion() {
        if (currentQuestionIndex > 0) {
            showQuestion(currentQuestionIndex - 1);
        }
    }
    
    // 计算测试结果
    private void calculateResult() {
        // 确保所有问题都已回答
        if (answers.size() < questions.size()) {
            Toast.makeText(getContext(), "请回答所有问题", Toast.LENGTH_SHORT).show();
            return;
        }
        
        // 计算原始总分
        int rawScore = 0;
        for (Integer score : answers.values()) {
            rawScore += score;
        }
        
        // 计算标准分（原始分乘以1.25）
        int standardScore = (int) (rawScore * 1.25);
        
        // 显示结果
        showResult(rawScore, standardScore);
        
        // 保存测试完成状态
        isTestCompleted = true;
        saveAnswersToLocal();
        
        // 确保重新测试按钮可见
        Button retestButton = getView().findViewById(R.id.retestButton);
        if (retestButton != null) {
            retestButton.setVisibility(View.VISIBLE);
        }
    }
    
    // 显示测试结果
    private void showResult(int rawScore, int standardScore) {
        // 隐藏问题布局，显示结果布局
        questionLayout.setVisibility(View.GONE);
        resultLayout.setVisibility(View.VISIBLE);
        
        // 显示总分
        totalScoreText.setText(String.format("原始总分: %d | 标准分: %d", rawScore, standardScore));
        
        // 根据标准分判断抑郁程度
        String level;
        String interpretation;
        
        if (standardScore < 50) {
            level = "正常范围";
            interpretation = "您的抑郁水平在正常范围内，没有明显的抑郁症状。";
        } else if (standardScore <= 59) {
            level = "轻度抑郁";
            interpretation = "您有轻度抑郁症状，建议关注自己的心理健康状况，学习一些放松技巧和压力管理方法。";
        } else if (standardScore <= 69) {
            level = "中度抑郁";
            interpretation = "您有中度抑郁症状，建议寻求专业心理咨询师的帮助，学习应对抑郁的方法。";
        } else {
            level = "重度抑郁";
            interpretation = "您有重度抑郁症状，强烈建议尽快寻求专业心理医生或精神科医生的帮助。";
        }
        
        resultLevelText.setText(String.format("抑郁程度: %s", level));
        resultInterpretationText.setText(interpretation);
    }
    
    // 重新开始测试
    private void restartTest() {
        // 清空答案
        answers.clear();
        currentQuestionIndex = 0;
        isTestCompleted = false;
        
        // 更新本地存储中的测试状态（不清除答案记录）
        SharedPreferences prefs = requireActivity().getSharedPreferences("user_info", Context.MODE_PRIVATE);
        prefs.edit()
            .putBoolean("sds_test_completed_" + userId, false)
            .putBoolean(PREF_SDS_IN_PROGRESS + "_" + userId, true)
            .apply();
        
        // 显示第一题
        questionLayout.setVisibility(View.VISIBLE);
        resultLayout.setVisibility(View.GONE);
        showQuestion(0);
    }
    
    // 清空所有作答
    private void clearAllAnswers() {
        // 清空内存中的作答
        answers.clear();
        currentQuestionIndex = 0;
        isTestCompleted = false;
        
        // 清空本地存储
        clearLocalAnswers();
        
        // 显示第一题并提示用户
        questionLayout.setVisibility(View.VISIBLE);
        resultLayout.setVisibility(View.GONE);
        showQuestion(0);
        Toast.makeText(getContext(), "已清空所有作答", Toast.LENGTH_SHORT).show();
    }
    
    // 清除本地存储的答案
    private void clearLocalAnswers() {
        SharedPreferences prefs = requireActivity().getSharedPreferences("user_info", Context.MODE_PRIVATE);
        prefs.edit()
            .remove(PREF_SDS_ANSWERS + "_" + userId)
            .remove(PREF_SDS_CURRENT_INDEX + "_" + userId)
            .remove(PREF_SDS_IN_PROGRESS + "_" + userId)
            .apply();
    }
    
    @Override
    public void onResume() {
        super.onResume();
        
        // 只有当问题列表已加载完毕时才恢复作答状态
        if (questions != null && !questions.isEmpty() && userId != -1) {
            restoreAnswersFromLocal();
        }
    }
    
    @Override
    public void onPause() {
        super.onPause();
        
        // 保存当前作答状态到本地
        if (userId != -1 && !answers.isEmpty()) {
            saveAnswersToLocal();
        }
    }
    
    // 保存当前作答到本地
    private void saveAnswersToLocal() {
        SharedPreferences prefs = requireActivity().getSharedPreferences("user_info", Context.MODE_PRIVATE);
        SharedPreferences.Editor editor = prefs.edit();
        
        // 使用Gson将Map转换为JSON字符串
        Gson gson = new Gson();
        String answersJson = gson.toJson(answers);
        
        // 保存当前作答状态
        editor.putString(PREF_SDS_ANSWERS + "_" + userId, answersJson);
        editor.putLong(PREF_SDS_CURRENT_INDEX + "_" + userId, currentQuestionIndex);
        editor.putBoolean(PREF_SDS_IN_PROGRESS + "_" + userId, questionLayout.getVisibility() == View.VISIBLE);
        editor.putBoolean("sds_test_completed_" + userId, isTestCompleted);
        
        // 如果测试已完成，保存测试结果
        if (isTestCompleted) {
            // 计算原始总分
            int rawScore = 0;
            for (Integer score : answers.values()) {
                rawScore += score;
            }
            
            // 计算标准分（原始分乘以1.25）
            int standardScore = (int) (rawScore * 1.25);
            
            // 保存分数 - 使用putLong避免类型不一致问题
            editor.putLong("sds_raw_score_" + userId, rawScore);
            editor.putLong("sds_standard_score_" + userId, standardScore);
        }
        
        editor.apply();
    }
    
    // 从本地恢复作答状态
    private void restoreAnswersFromLocal() {
        SharedPreferences prefs = requireActivity().getSharedPreferences("user_info", Context.MODE_PRIVATE);
        
        // 检查测试是否已完成
        boolean testCompleted = prefs.getBoolean("sds_test_completed_" + userId, false);
        
        // 恢复答案
        String answersJson = prefs.getString(PREF_SDS_ANSWERS + "_" + userId, "");
        if (!answersJson.isEmpty()) {
            Gson gson = new Gson();
            Type type = new TypeToken<Map<Integer, Integer>>(){}.getType();
            answers = gson.fromJson(answersJson, type);
        }
        
        if (testCompleted) {
            // 测试已完成，显示结果
            isTestCompleted = true;
            
            // 获取保存的分数 - 使用getLong并转为int，避免类型转换异常
            int rawScore = (int) prefs.getLong("sds_raw_score_" + userId, 0);
            int standardScore = (int) prefs.getLong("sds_standard_score_" + userId, 0);
            
            // 显示结果
            questionLayout.setVisibility(View.GONE);
            resultLayout.setVisibility(View.VISIBLE);
            progressBar.setVisibility(View.GONE);
            
            // 显示分数和解释
            showResult(rawScore, standardScore);
            
            // 修改重新测试按钮文本和点击事件
             Button retestButton = getView().findViewById(R.id.retestButton);
             if (retestButton != null) {
                 retestButton.setText("重新测试");
                 retestButton.setOnClickListener(v -> restartTest());
                 retestButton.setVisibility(View.VISIBLE);
             }
        } else {
            // 检查是否有进行中的测试
            boolean inProgress = prefs.getBoolean(PREF_SDS_IN_PROGRESS + "_" + userId, false);
            
            if (inProgress && !answers.isEmpty()) {
                // 恢复当前问题索引 - 使用getLong并转为int，避免类型转换异常
                currentQuestionIndex = (int) prefs.getLong(PREF_SDS_CURRENT_INDEX + "_" + userId, 0);
                
                // 显示当前问题
                questionLayout.setVisibility(View.VISIBLE);
                resultLayout.setVisibility(View.GONE);
                progressBar.setVisibility(View.GONE);
                showQuestion(currentQuestionIndex);
            } else {
                // 没有进行中的测试，显示第一题
                questionLayout.setVisibility(View.VISIBLE);
                resultLayout.setVisibility(View.GONE);
                progressBar.setVisibility(View.GONE);
                showQuestion(0);
            }
        }
    }
    
    // SDS问题类
    public static class SDSQuestion {
        private int id;
        private String content;
        private boolean reversed; // 是否反向计分
        
        public SDSQuestion(int id, String content) {
            this.id = id;
            this.content = content;
            this.reversed = false;
        }
        
        public SDSQuestion(int id, String content, boolean reversed) {
            this.id = id;
            this.content = content;
            this.reversed = reversed;
        }
        
        public int getId() {
            return id;
        }
        
        public String getContent() {
            return content;
        }
        
        public boolean isReversed() {
            return reversed;
        }
    }
}