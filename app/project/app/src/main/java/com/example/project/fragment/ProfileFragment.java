package com.example.project.fragment;

import android.Manifest;
import android.app.Activity;
import android.app.AlertDialog;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.drawable.Drawable;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.provider.MediaStore;
import android.text.InputType;
import android.text.TextUtils;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.cardview.widget.CardView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.fragment.app.Fragment;
import androidx.localbroadcastmanager.content.LocalBroadcastManager;

import com.bumptech.glide.Glide;
import com.bumptech.glide.load.DataSource;
import com.bumptech.glide.load.engine.GlideException;
import com.bumptech.glide.request.RequestListener;
import com.bumptech.glide.request.target.Target;
import com.example.project.LLamaAPI;
import com.example.project.LoginActivity;
// 不再需要ModelDownloadService导入，因为我们从assets加载模型
import com.example.project.R;
import com.example.project.api.ApiClient;
import com.example.project.model.User;
import com.example.project.service.PsychologicalStatusService;
import com.example.project.utils.PsychologicalStatusManager;
import com.example.project.utils.ModelLoadingDialogManager;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.TimeUnit;

import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;
import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.RequestBody;
// 不再需要其他OkHttp相关导入，因为我们从assets加载模型

public class ProfileFragment extends Fragment implements LLamaAPI.ModelStateListener {
    private static final String TAG = "ProfileFragment";
    
    private CardView avatarContainer;
    private ImageView avatarImage;
    private TextView usernameText;
    private TextView mbtiTypeText;
    private TextView bioText;
    private TextView gradeText;
    private TextView genderText;
    private TextView ageText;
    private View gradeContainer;
    private View genderContainer;
    private View ageContainer;
    private TextView changePasswordButton;
    private TextView logoutButton;

    // 添加心理状态显示相关UI元素
    private TextView depressionLevelText;
    private TextView anxietyLevelText;
    private TextView lastAssessmentText;

    // 添加模型管理相关UI元素
    private CardView modelManagementCard;
    private TextView modelStatusText;
    private Button loadModelButton;
    private ProgressBar modelLoadingProgress;
    
    private Long userId;
    private User currentUser;
    private LLamaAPI profileLlamaApi; // 专用于个人中心的LLamaAPI实例
    private LLamaAPI statusLlamaApi; // 专用于状态判断的LLamaAPI实例
    private boolean isModelLoading = false;
    private PsychologicalStatusService psychologicalStatusService;
    
    // 模型相关变量
    private static final String MODEL_FILENAME = "Bluecat_chat.gguf"; // 模型文件名，与assets/models目录中的文件名一致

    private static final int PICK_IMAGE_REQUEST = 1;
    private static final int PERMISSION_REQUEST = 2;
    
    // 不再需要广播接收器，因为我们从assets加载模型

    public static ProfileFragment newInstance() {
        return new ProfileFragment();
    }

    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container,
                           Bundle savedInstanceState) {
        return inflater.inflate(R.layout.fragment_profile, container, false);
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);
        initViews(view);
        setupListeners();
        loadUserInfo();
        
        // 使用静态变量保存LLamaAPI实例，避免在页面切换时重新加载模型
        if (sharedProfileLlamaApi == null) {
            sharedProfileLlamaApi = LLamaAPI.createInstance("profile_model");
            Log.d(TAG, "创建个人中心专用LLamaAPI实例");
        }
        profileLlamaApi = sharedProfileLlamaApi;
        
        // 使用静态变量保存状态判断模型的LLamaAPI实例
        if (sharedStatusLlamaApi == null) {
            sharedStatusLlamaApi = LLamaAPI.createInstance("status_model");
            Log.d(TAG, "创建状态判断专用LLamaAPI实例");
        }
        statusLlamaApi = sharedStatusLlamaApi;
        
        // 初始化心理状态评估服务并传递状态判断模型实例
        psychologicalStatusService = new PsychologicalStatusService(requireContext());
        psychologicalStatusService.setStatusLlamaAPI(statusLlamaApi);
        Log.d(TAG, "已将状态判断模型实例传递给心理状态评估服务");
        
        // 不再需要OkHttpClient，因为不再下载模型
        
        // 检查模型状态，但不自动加载
        updateModelStatus();
    }

    @Override
    public void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == PICK_IMAGE_REQUEST && resultCode == Activity.RESULT_OK && data != null && data.getData() != null) {
            uploadImage(data.getData());
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == PERMISSION_REQUEST) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                openImagePicker();
            } else if (isAdded() && getContext() != null) {
                Toast.makeText(requireContext(), "需要存储权限才能选择图片", Toast.LENGTH_SHORT).show();
            }
        }
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
        // 不再需要注销广播接收器，因为我们从assets加载模型
        // 不卸载模型，保留静态实例以便在Fragment之间共享
        Log.d(TAG, "ProfileFragment销毁，但保留模型实例以便重用");
    }

    private void initViews(View view) {
        avatarContainer = view.findViewById(R.id.avatarContainer);
        avatarImage = view.findViewById(R.id.avatarImage);
        usernameText = view.findViewById(R.id.usernameText);
        mbtiTypeText = view.findViewById(R.id.mbtiTypeText);
        bioText = view.findViewById(R.id.bioText);
        gradeText = view.findViewById(R.id.gradeText);
        genderText = view.findViewById(R.id.genderText);
        ageText = view.findViewById(R.id.ageText);
        gradeContainer = view.findViewById(R.id.gradeContainer);
        genderContainer = view.findViewById(R.id.genderContainer);
        ageContainer = view.findViewById(R.id.ageContainer);
        changePasswordButton = view.findViewById(R.id.changePasswordButton);
        logoutButton = view.findViewById(R.id.logoutButton);

        // 初始化心理状态显示相关UI元素
        depressionLevelText = view.findViewById(R.id.depressionText);
        anxietyLevelText = view.findViewById(R.id.anxietyText);
        lastAssessmentText = view.findViewById(R.id.lastAssessmentText);

        // 初始化模型管理相关UI元素
        modelManagementCard = view.findViewById(R.id.modelManagementCard);
        modelStatusText = view.findViewById(R.id.modelStatusText);
        loadModelButton = view.findViewById(R.id.loadModelButton);
        modelLoadingProgress = view.findViewById(R.id.modelLoadingProgress);

        // 从SharedPreferences获取用户ID
        SharedPreferences prefs = requireActivity().getSharedPreferences("user_info", Context.MODE_PRIVATE);
        userId = prefs.getLong("user_id", -1);
    }

    private void setupListeners() {
        avatarContainer.setOnClickListener(v -> showChangeAvatarDialog());
        usernameText.setOnClickListener(v -> showEditUsernameDialog());
        bioText.setOnClickListener(v -> showEditBioDialog());
        gradeContainer.setOnClickListener(v -> showEditGradeDialog());
        genderContainer.setOnClickListener(v -> showEditGenderDialog());
        ageContainer.setOnClickListener(v -> showEditAgeDialog());
        changePasswordButton.setOnClickListener(v -> showChangePasswordDialog());
        logoutButton.setOnClickListener(v -> showLogoutConfirmDialog());
        
        // 设置模型管理相关监听器
        loadModelButton.setOnClickListener(v -> loadModel());
        
        // 隐藏模型管理卡片，因为现在通过AI心理咨询页面直接加载模型
        if (modelManagementCard != null) {
            modelManagementCard.setVisibility(View.GONE);
        }
    }

    private void loadUserInfo() {
        if (userId == -1) {
            if (isAdded() && getContext() != null) {
                Toast.makeText(getContext(), "请先登录", Toast.LENGTH_SHORT).show();
            }
            return;
        }

        ApiClient.getUserApi().getUserInfo(userId).enqueue(new Callback<User>() {
            @Override
            public void onResponse(Call<User> call, Response<User> response) {
                if (isAdded()) {
                    if (response.isSuccessful() && response.body() != null) {
                        currentUser = response.body();
                        updateUI();
                    }
                }
            }

            @Override
            public void onFailure(Call<User> call, Throwable t) {
                if (isAdded() && getContext() != null) {
                    Toast.makeText(getContext(), "获取用户信息失败: " + t.getMessage(), Toast.LENGTH_SHORT).show();
                }
            }
        });
    }

    private void updateUI() {
        if (currentUser == null) return;

        usernameText.setText(currentUser.getUsername());
        // 隐藏MBTI类型显示
        mbtiTypeText.setVisibility(View.GONE);
        bioText.setText(currentUser.getBio() != null ? currentUser.getBio() : "点击添加个性签名");
        gradeText.setText(currentUser.getGrade() != null ? currentUser.getGrade() : "未设置");
        genderText.setText(currentUser.getGender() != null ? currentUser.getGender() : "未设置");
        ageText.setText(currentUser.getAge() != null ? String.valueOf(currentUser.getAge()) : "未设置");

        // 加载头像
        if (currentUser.getAvatarUrl() != null) {
            String avatarUrl = ApiClient.BASE_URL.substring(0, ApiClient.BASE_URL.length() - 1) + currentUser.getAvatarUrl();
            Log.d(TAG, "Loading avatar from URL: " + avatarUrl);
            
            Glide.with(this)
                    .load(avatarUrl)
                    .placeholder(R.drawable.default_avatar)
                    .error(R.drawable.default_avatar)
                    .listener(new RequestListener<Drawable>() {
                        @Override
                        public boolean onLoadFailed(@Nullable GlideException e, Object model,
                                                  Target<Drawable> target, boolean isFirstResource) {
                            Log.e(TAG, "Avatar load failed for URL: " + avatarUrl + ", error: " + 
                                  (e != null ? e.getMessage() : "unknown"));
                            return false;
                        }

                        @Override
                        public boolean onResourceReady(Drawable resource, Object model,
                                                     Target<Drawable> target, DataSource dataSource,
                                                     boolean isFirstResource) {
                            Log.d(TAG, "Avatar loaded successfully from: " + avatarUrl);
                            return false;
                        }
                    })
                    .into(avatarImage);
        } else {
            avatarImage.setImageResource(R.drawable.default_avatar);
        }
        
        // 加载心理状态数据
        loadPsychologicalStatus();
    }

    private void loadPsychologicalStatus() {
        if (getContext() == null) return;
        
        try {
            // 获取最新的心理状态评估结果
            Map<String, Object> latestStatus = PsychologicalStatusManager.getLatestStatusResult(getContext());
            
            if (latestStatus != null) {
                // 获取抑郁和焦虑程度
                int depression = (Integer) latestStatus.get("depression");
                int anxiety = (Integer) latestStatus.get("anxiety");
                String timestamp = (String) latestStatus.get("timestamp");
                
                // 转换数字等级为文字描述
                String depressionLevel = getStatusLevelText(depression);
                String anxietyLevel = getStatusLevelText(anxiety);
                
                // 更新UI显示
                depressionLevelText.setText(depressionLevel);
                anxietyLevelText.setText(anxietyLevel);
                
                // 格式化时间显示
                if (timestamp != null && !timestamp.isEmpty()) {
                    lastAssessmentText.setText(formatTimestamp(timestamp));
                } else {
                    lastAssessmentText.setText("暂无记录");
                }
            } else {
                // 没有评估记录时的默认显示
                depressionLevelText.setText("暂无评估");
                anxietyLevelText.setText("暂无评估");
                lastAssessmentText.setText("暂无记录");
            }
        } catch (Exception e) {
            Log.e(TAG, "加载心理状态数据失败", e);
            // 出错时的默认显示
            depressionLevelText.setText("加载失败");
            anxietyLevelText.setText("加载失败");
            lastAssessmentText.setText("加载失败");
        }
    }
    
    private String getStatusLevelText(int level) {
        switch (level) {
            case 0:
                return "无症状";
            case 1:
                return "轻度";
            case 2:
                return "中度";
            case 3:
                return "重度";
            default:
                return "未知";
        }
    }
    
    private String formatTimestamp(String timestamp) {
        try {
            // 假设timestamp格式为 "yyyy-MM-dd HH:mm:ss"
            // 这里简化处理，只显示日期部分
            if (timestamp.length() >= 10) {
                return timestamp.substring(0, 10);
            }
            return timestamp;
        } catch (Exception e) {
            return "格式错误";
        }
    }

    private void checkStoragePermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            // Android 10及以上版本不需要存储权限就可以访问媒体文件
            openImagePicker();
        } else {
            // Android 9及以下版本需要请求存储权限
            if (ContextCompat.checkSelfPermission(requireContext(), Manifest.permission.READ_EXTERNAL_STORAGE)
                    != PackageManager.PERMISSION_GRANTED) {
                ActivityCompat.requestPermissions(requireActivity(),
                        new String[]{Manifest.permission.READ_EXTERNAL_STORAGE},
                        PERMISSION_REQUEST);
            } else {
                openImagePicker();
            }
        }
    }

    private void showChangeAvatarDialog() {
        String[] options = {"从相册选择", "取消"};
        new AlertDialog.Builder(requireContext())
                .setTitle("更换头像")
                .setItems(options, (dialog, which) -> {
                    if (which == 0) {
                        checkStoragePermission();
                    }
                })
                .show();
    }

    private void openImagePicker() {
        Intent intent = new Intent();
        intent.setType("image/*");
        intent.setAction(Intent.ACTION_GET_CONTENT);
        startActivityForResult(Intent.createChooser(intent, "选择图片"), PICK_IMAGE_REQUEST);
    }

    private void uploadImage(Uri imageUri) {
        try {
            // 获取原始图片
            Bitmap originalBitmap = MediaStore.Images.Media.getBitmap(requireContext().getContentResolver(), imageUri);
            
            // 压缩图片
            Bitmap compressedBitmap = compressImage(originalBitmap);
            
            // 确保缓存目录存在
            File cacheDir = requireContext().getCacheDir();
            if (!cacheDir.exists()) {
                cacheDir.mkdirs();
            }
            
            // 将压缩后的图片转换为文件
            File compressedFile = new File(cacheDir, "compressed_avatar.jpg");
            FileOutputStream fos = new FileOutputStream(compressedFile);
            compressedBitmap.compress(Bitmap.CompressFormat.JPEG, 80, fos);
            fos.flush();
            fos.close();
            
            Log.d(TAG, "压缩后的图片已保存到: " + compressedFile.getAbsolutePath() + ", 文件大小: " + compressedFile.length() + " 字节");

            // 直接使用文件路径创建RequestBody
            RequestBody requestFile = RequestBody.create(MediaType.parse("image/jpeg"), compressedFile);
            MultipartBody.Part body = MultipartBody.Part.createFormData("file", "image.jpg", requestFile);
            RequestBody userId = RequestBody.create(MediaType.parse("text/plain"), String.valueOf(this.userId));

            // 发送请求
            ApiClient.getUserApi().uploadAvatar(body, userId).enqueue(new Callback<Map<String, String>>() {
                @Override
                public void onResponse(Call<Map<String, String>> call, Response<Map<String, String>> response) {
                    if (isAdded() && getContext() != null) {
                        if (response.isSuccessful() && response.body() != null) {
                            String avatarUrl = response.body().get("url");
                            Log.d(TAG, "头像上传成功，URL: " + avatarUrl);
                            
                            if (currentUser != null) {
                                currentUser.setAvatarUrl(avatarUrl);
                            }
                            
                            // 更新UI显示新头像
                            String fullUrl = ApiClient.BASE_URL.substring(0, ApiClient.BASE_URL.length() - 1) + avatarUrl;
                            Log.d(TAG, "加载头像: " + fullUrl);
                            
                            Glide.with(ProfileFragment.this)
                                    .load(fullUrl)
                                    .placeholder(R.drawable.default_avatar)
                                    .error(R.drawable.default_avatar)
                                    .into(avatarImage);
                            Toast.makeText(getContext(), "头像上传成功", Toast.LENGTH_SHORT).show();
                        } else {
                            try {
                                String errorBody = response.errorBody() != null ? response.errorBody().string() : "Unknown error";
                                Log.e(TAG, "上传失败: HTTP " + response.code() + " - " + errorBody);
                                Toast.makeText(getContext(), "头像上传失败: " + errorBody, Toast.LENGTH_SHORT).show();
                            } catch (IOException e) {
                                Log.e(TAG, "读取错误响应失败", e);
                                Toast.makeText(getContext(), "头像上传失败", Toast.LENGTH_SHORT).show();
                            }
                        }
                    }
                }

                @Override
                public void onFailure(Call<Map<String, String>> call, Throwable t) {
                    if (isAdded() && getContext() != null) {
                        Log.e(TAG, "上传请求失败", t);
                        Toast.makeText(getContext(), "头像上传失败: " + t.getMessage(), Toast.LENGTH_SHORT).show();
                    }
                }
            });

            // 清理资源
            originalBitmap.recycle();
            compressedBitmap.recycle();
            // 不要立即删除文件，等上传完成后再删除
            // compressedFile.delete();

        } catch (Exception e) {
            Log.e(TAG, "处理图片失败", e);
            if (isAdded() && getContext() != null) {
                Toast.makeText(getContext(), "文件处理失败: " + e.getMessage(), Toast.LENGTH_SHORT).show();
            }
        }
    }

    private Bitmap compressImage(Bitmap image) {
        if (image == null) return null;

        // 计算压缩比例
        int originalWidth = image.getWidth();
        int originalHeight = image.getHeight();
        
        // 目标大小为800像素（可以根据需求调整）
        float maxDimension = 800.0f;
        
        float scale = 1.0f;
        if (originalWidth > originalHeight) {
            if (originalWidth > maxDimension) {
                scale = maxDimension / originalWidth;
            }
        } else {
            if (originalHeight > maxDimension) {
                scale = maxDimension / originalHeight;
            }
        }
        
        // 计算新的尺寸
        int newWidth = Math.round(originalWidth * scale);
        int newHeight = Math.round(originalHeight * scale);
        
        // 创建新的缩放后的位图
        return Bitmap.createScaledBitmap(image, newWidth, newHeight, true);
    }

    private void showEditUsernameDialog() {
        if (!isAdded() || getContext() == null) return;
        
        EditText input = new EditText(getContext());
        input.setText(currentUser.getUsername());
        input.setInputType(InputType.TYPE_CLASS_TEXT);
        input.setSingleLine(true);

        new AlertDialog.Builder(getContext())
                .setTitle("修改用户名")
                .setView(input)
                .setPositiveButton("确定", (dialog, which) -> {
                    String newUsername = input.getText().toString().trim();
                    if (newUsername.isEmpty()) {
                        if (isAdded() && getContext() != null) {
                            Toast.makeText(getContext(), "用户名不能为空", Toast.LENGTH_SHORT).show();
                        }
                        return;
                    }
                    if (newUsername.equals(currentUser.getUsername())) {
                        return;
                    }
                    updateUserField("username", newUsername);
                })
                .setNegativeButton("取消", null)
                .show();
    }

    private void showEditBioDialog() {
        AlertDialog.Builder builder = new AlertDialog.Builder(requireContext());
        builder.setTitle("编辑个性签名");

        final EditText input = new EditText(requireContext());
        input.setText(currentUser.getBio());
        builder.setView(input);

        builder.setPositiveButton("确定", (dialog, which) -> {
            String newBio = input.getText().toString().trim();
            updateUserField("bio", newBio);
        });
        builder.setNegativeButton("取消", null);

        builder.show();
    }

    private void showEditGradeDialog() {
        AlertDialog.Builder builder = new AlertDialog.Builder(requireContext());
        builder.setTitle("选择年级");

        final String[] grades = {"大一", "大二", "大三", "大四", "研究生"};
        builder.setItems(grades, (dialog, which) -> {
            String selectedGrade = grades[which];
            updateUserField("grade", selectedGrade);
        });

        builder.show();
    }

    private void showEditGenderDialog() {
        AlertDialog.Builder builder = new AlertDialog.Builder(requireContext());
        builder.setTitle("选择性别");

        final String[] genders = {"男", "女", "其他"};
        builder.setItems(genders, (dialog, which) -> {
            String selectedGender = genders[which];
            updateUserField("gender", selectedGender);
        });

        builder.show();
    }

    private void showEditAgeDialog() {
        AlertDialog.Builder builder = new AlertDialog.Builder(requireContext());
        builder.setTitle("编辑年龄");

        final EditText input = new EditText(requireContext());
        input.setInputType(android.text.InputType.TYPE_CLASS_NUMBER);
        if (currentUser.getAge() != null) {
            input.setText(String.valueOf(currentUser.getAge()));
        }
        builder.setView(input);

        builder.setPositiveButton("确定", (dialog, which) -> {
            try {
                int newAge = Integer.parseInt(input.getText().toString().trim());
                if (newAge > 0 && newAge < 150) {
                    updateUserField("age", String.valueOf(newAge));
                } else {
                    Toast.makeText(getContext(), "请输入有效年龄", Toast.LENGTH_SHORT).show();
                }
            } catch (NumberFormatException e) {
                Toast.makeText(getContext(), "请输入有效数字", Toast.LENGTH_SHORT).show();
            }
        });
        builder.setNegativeButton("取消", null);

        builder.show();
    }

    private void showChangePasswordDialog() {
        View dialogView = LayoutInflater.from(requireContext()).inflate(R.layout.dialog_change_password, null);
        EditText oldPasswordInput = dialogView.findViewById(R.id.oldPasswordInput);
        EditText newPasswordInput = dialogView.findViewById(R.id.newPasswordInput);
        EditText confirmPasswordInput = dialogView.findViewById(R.id.confirmPasswordInput);

        AlertDialog.Builder builder = new AlertDialog.Builder(requireContext());
        builder.setTitle("修改密码")
                .setView(dialogView)
                .setPositiveButton("确定", (dialog, which) -> {
                    String oldPassword = oldPasswordInput.getText().toString().trim();
                    String newPassword = newPasswordInput.getText().toString().trim();
                    String confirmPassword = confirmPasswordInput.getText().toString().trim();

                    if (oldPassword.isEmpty() || newPassword.isEmpty() || confirmPassword.isEmpty()) {
                        Toast.makeText(getContext(), "请填写所有字段", Toast.LENGTH_SHORT).show();
                        return;
                    }

                    if (!newPassword.equals(confirmPassword)) {
                        Toast.makeText(getContext(), "新密码两次输入不一致", Toast.LENGTH_SHORT).show();
                        return;
                    }

                    updatePassword(oldPassword, newPassword);
                })
                .setNegativeButton("取消", null);

        builder.show();
    }

    private void showLogoutConfirmDialog() {
        new AlertDialog.Builder(requireContext())
                .setTitle("退出登录")
                .setMessage("确定要退出登录吗？")
                .setPositiveButton("确定", (dialog, which) -> logout())
                .setNegativeButton("取消", null)
                .show();
    }

    private void updateUserField(String field, String value) {
        Map<String, String> updateData = new HashMap<>();
        updateData.put(field, value);

        ApiClient.getUserApi().updateUserField(userId, updateData).enqueue(new Callback<User>() {
            @Override
            public void onResponse(Call<User> call, Response<User> response) {
                if (isAdded() && getContext() != null) {
                    if (response.isSuccessful() && response.body() != null) {
                        currentUser = response.body();
                        updateUI();
                        Toast.makeText(getContext(), "更新成功", Toast.LENGTH_SHORT).show();
                    } else {
                        Toast.makeText(getContext(), "更新失败", Toast.LENGTH_SHORT).show();
                    }
                }
            }

            @Override
            public void onFailure(Call<User> call, Throwable t) {
                if (isAdded() && getContext() != null) {
                    Toast.makeText(getContext(), "更新失败: " + t.getMessage(), Toast.LENGTH_SHORT).show();
                }
            }
        });
    }

    private void updatePassword(String oldPassword, String newPassword) {
        Map<String, String> passwordData = new HashMap<>();
        passwordData.put("oldPassword", oldPassword);
        passwordData.put("newPassword", newPassword);

        ApiClient.getUserApi().updatePassword(userId, passwordData).enqueue(new Callback<Void>() {
            @Override
            public void onResponse(Call<Void> call, Response<Void> response) {
                if (isAdded() && getContext() != null) {
                    if (response.isSuccessful()) {
                        Toast.makeText(getContext(), "密码修改成功", Toast.LENGTH_SHORT).show();
                    } else {
                        Toast.makeText(getContext(), "密码修改失败，请检查原密码是否正确", Toast.LENGTH_SHORT).show();
                    }
                }
            }

            @Override
            public void onFailure(Call<Void> call, Throwable t) {
                if (isAdded() && getContext() != null) {
                    Toast.makeText(getContext(), "密码修改失败: " + t.getMessage(), Toast.LENGTH_SHORT).show();
                }
            }
        });
    }

    private void logout() {
        // 清除SharedPreferences中的用户信息
        SharedPreferences.Editor editor = requireActivity().getSharedPreferences("user_info", Context.MODE_PRIVATE).edit();
        editor.clear();
        editor.apply();

        // 跳转到登录页面
        Intent intent = new Intent(requireActivity(), LoginActivity.class);
        intent.setFlags(Intent.FLAG_ACTIVITY_NEW_TASK | Intent.FLAG_ACTIVITY_CLEAR_TASK);
        startActivity(intent);
        requireActivity().finish();
    }

    // 从assets目录加载模型文件
    private void loadModel() {
        if (isModelLoading) {
            if (isAdded() && getContext() != null) {
                Toast.makeText(getContext(), "模型正在加载中，请稍候...", Toast.LENGTH_SHORT).show();
            }
            return;
        }
        
        // 获取聊天模型文件路径
        String chatModelFilePath = getModelFilePath();
        File chatModelFile = new File(chatModelFilePath);
        
        // 获取状态判断模型文件路径
        String statusModelFilename = "Bluecat_status.gguf";
        File filesDir = getContext().getFilesDir();
        File modelsDir = new File(filesDir, "models");
        String statusModelFilePath = new File(modelsDir, statusModelFilename).getAbsolutePath();
        File statusModelFile = new File(statusModelFilePath);
        
        Log.d(TAG, "聊天模型路径: " + chatModelFilePath);
        Log.d(TAG, "聊天模型文件存在: " + chatModelFile.exists());
        if (chatModelFile.exists()) {
            Log.d(TAG, "聊天模型文件大小: " + chatModelFile.length() + " bytes");
            Log.d(TAG, "聊天模型文件可读: " + chatModelFile.canRead());
        }
        
        Log.d(TAG, "状态判断模型路径: " + statusModelFilePath);
        Log.d(TAG, "状态判断模型文件存在: " + statusModelFile.exists());
        if (statusModelFile.exists()) {
            Log.d(TAG, "状态判断模型文件大小: " + statusModelFile.length() + " bytes");
            Log.d(TAG, "状态判断模型文件可读: " + statusModelFile.canRead());
        }
        
        // 检查两个模型文件的状态
        boolean chatModelReady = chatModelFile.exists() && chatModelFile.length() >= 10 * 1024 * 1024 && chatModelFile.canRead(); // 确保文件至少有10MB且可读
        boolean statusModelReady = statusModelFile.exists() && statusModelFile.length() >= 10 * 1024 * 1024 && statusModelFile.canRead();
        
        // 根据模型文件状态决定下一步操作
        if (!chatModelReady) {
            try {
                // 确保目录存在
                File parentDir = chatModelFile.getParentFile();
                if (parentDir != null && !parentDir.exists()) {
                    boolean created = parentDir.mkdirs();
                    if (!created) {
                        throw new IOException("无法创建目录: " + parentDir.getAbsolutePath());
                    }
                }
                
                // 从assets复制聊天模型文件
                copyModelFromAssets();
            } catch (IOException e) {
                Log.e(TAG, "从assets复制聊天模型文件失败", e);
                if (isAdded() && getContext() != null) {
                    Toast.makeText(getContext(), "聊天模型文件准备失败: " + e.getMessage(), Toast.LENGTH_SHORT).show();
                    modelStatusText.setText("聊天模型文件准备失败");
                }
                return;
            }
        } else if (!statusModelReady) {
            try {
                // 确保目录存在
                File parentDir = statusModelFile.getParentFile();
                if (parentDir != null && !parentDir.exists()) {
                    boolean created = parentDir.mkdirs();
                    if (!created) {
                        throw new IOException("无法创建目录: " + parentDir.getAbsolutePath());
                    }
                }
                
                // 从assets复制状态判断模型文件
                copyStatusModelFromAssets(statusModelFilename);
            } catch (IOException e) {
                Log.e(TAG, "从assets复制状态判断模型文件失败", e);
                if (isAdded() && getContext() != null) {
                    Toast.makeText(getContext(), "状态判断模型文件准备失败: " + e.getMessage(), Toast.LENGTH_SHORT).show();
                    modelStatusText.setText("状态判断模型文件准备失败");
                }
                return;
            }
        } else {
            // 两个模型文件都已准备好，加载聊天模型
            loadModelFromFile(chatModelFilePath);
        }
    }
    
    // 不再需要下载模型，因为使用assets目录中的模型文件
    
    private String getModelFilePath() {
        if (!isAdded() || getContext() == null) return "";
        // 使用应用私有存储目录，这样在卸载应用时会自动清理
        File filesDir = getContext().getFilesDir();
        File modelsDir = new File(filesDir, "models");
        return new File(modelsDir, MODEL_FILENAME).getAbsolutePath();
    }
    
    // 从assets目录复制聊天模型文件到应用私有存储
    private void copyModelFromAssets() throws IOException {
        if (!isAdded() || getContext() == null) throw new IOException("Context不可用");
        
        String modelFilePath = getModelFilePath();
        File modelFile = new File(modelFilePath);
        
        // 确保目标目录存在
        File parentDir = modelFile.getParentFile();
        if (parentDir != null && !parentDir.exists()) {
            boolean created = parentDir.mkdirs();
            if (!created) {
                throw new IOException("无法创建目录: " + parentDir.getAbsolutePath());
            }
        }
        
        // 显示复制进度
        modelLoadingProgress.setVisibility(View.VISIBLE);
        modelStatusText.setText("正在准备聊天模型文件...");
        loadModelButton.setEnabled(false);
        isModelLoading = true;
        
        // 从assets目录复制模型文件
        try (InputStream in = getContext().getAssets().open("models/" + MODEL_FILENAME);
             OutputStream out = new FileOutputStream(modelFile)) {
            
            byte[] buffer = new byte[8192];
            int read;
            long total = 0;
            long fileSize = in.available();
            
            Log.d(TAG, "开始从assets复制聊天模型文件，大小: " + fileSize + " 字节");
            
            while ((read = in.read(buffer)) != -1) {
                out.write(buffer, 0, read);
                total += read;
                
                // 更新进度
                final int progress = (int) ((total * 100) / fileSize);
                if (getActivity() != null) {
                    getActivity().runOnUiThread(() -> {
                        modelLoadingProgress.setProgress(progress);
                        modelStatusText.setText(String.format("准备聊天模型文件: %d%%", progress));
                    });
                }
            }
            
            Log.d(TAG, "聊天模型文件复制完成: " + modelFilePath + ", 大小: " + modelFile.length() + " 字节");
        } catch (Exception e) {
            isModelLoading = false;
            Log.e(TAG, "复制聊天模型文件失败", e);
            throw e;
        }
    }
    
    // 从assets目录复制状态判断模型文件到应用私有存储
    private void copyStatusModelFromAssets(String statusModelFilename) throws IOException {
        if (!isAdded() || getContext() == null) throw new IOException("Context不可用");
        
        File filesDir = getContext().getFilesDir();
        File modelsDir = new File(filesDir, "models");
        String statusModelFilePath = new File(modelsDir, statusModelFilename).getAbsolutePath();
        File statusModelFile = new File(statusModelFilePath);
        
        // 确保目标目录存在
        File parentDir = statusModelFile.getParentFile();
        if (parentDir != null && !parentDir.exists()) {
            boolean created = parentDir.mkdirs();
            if (!created) {
                throw new IOException("无法创建目录: " + parentDir.getAbsolutePath());
            }
        }
        
        // 显示复制进度
        modelLoadingProgress.setVisibility(View.VISIBLE);
        modelStatusText.setText("正在准备状态判断模型文件...");
        loadModelButton.setEnabled(false);
        isModelLoading = true;
        
        // 从assets目录复制模型文件
        try (InputStream in = getContext().getAssets().open("models/" + statusModelFilename);
             OutputStream out = new FileOutputStream(statusModelFile)) {
            
            byte[] buffer = new byte[8192];
            int read;
            long total = 0;
            long fileSize = in.available();
            
            Log.d(TAG, "开始从assets复制状态判断模型文件，大小: " + fileSize + " 字节");
            
            while ((read = in.read(buffer)) != -1) {
                out.write(buffer, 0, read);
                total += read;
                
                // 更新进度
                final int progress = (int) ((total * 100) / fileSize);
                if (getActivity() != null) {
                    getActivity().runOnUiThread(() -> {
                        modelLoadingProgress.setProgress(progress);
                        modelStatusText.setText(String.format("准备状态判断模型文件: %d%%", progress));
                    });
                }
            }
            
            Log.d(TAG, "状态判断模型文件复制完成: " + statusModelFilePath + ", 大小: " + statusModelFile.length() + " 字节");
            Log.d(TAG, "状态判断模型文件可读: " + statusModelFile.canRead());
            
            // 验证文件权限
            if (!statusModelFile.canRead()) {
                throw new IOException("状态判断模型文件无法读取: " + statusModelFilePath);
            }
        } catch (Exception e) {
            isModelLoading = false;
            Log.e(TAG, "复制状态判断模型文件失败", e);
            throw e;
        }
    }
    
    private void loadModelFromFile(String modelPath) {
        if (!isAdded() || getActivity() == null) return;
        
        // 检查聊天模型文件是否存在
        File modelFile = new File(modelPath);
        if (!modelFile.exists() || modelFile.length() == 0 || !modelFile.canRead()) {
            Log.e(TAG, "聊天模型文件不存在、为空或无法读取: " + modelPath);
            Log.e(TAG, "文件存在: " + modelFile.exists() + ", 文件大小: " + modelFile.length() + ", 文件可读: " + modelFile.canRead());
            if (isAdded() && getActivity() != null) {
                getActivity().runOnUiThread(() -> {
                    isModelLoading = false;
                    modelLoadingProgress.setVisibility(View.GONE);
                    modelStatusText.setText("聊天模型文件不存在、已损坏或无法读取");
                    loadModelButton.setEnabled(true);
                    if (getContext() != null) {
                        Toast.makeText(getContext(), "聊天模型文件不存在、已损坏或无法读取", Toast.LENGTH_SHORT).show();
                    }
                });
            }
            return;
        }
        
        // 检查状态判断模型文件是否存在
        String statusModelFilename = "Bluecat_status.gguf";
        File filesDir = getContext().getFilesDir();
        File modelsDir = new File(filesDir, "models");
        String statusModelFilePath = new File(modelsDir, statusModelFilename).getAbsolutePath();
        File statusModelFile = new File(statusModelFilePath);
        
        if (!statusModelFile.exists() || statusModelFile.length() == 0 || !statusModelFile.canRead()) {
            Log.e(TAG, "状态判断模型文件不存在、为空或无法读取: " + statusModelFilePath);
            Log.e(TAG, "文件存在: " + statusModelFile.exists() + ", 文件大小: " + statusModelFile.length() + ", 文件可读: " + statusModelFile.canRead());
            if (isAdded() && getActivity() != null) {
                getActivity().runOnUiThread(() -> {
                    isModelLoading = false;
                    modelLoadingProgress.setVisibility(View.GONE);
                    modelStatusText.setText("状态判断模型文件不存在、已损坏或无法读取");
                    loadModelButton.setEnabled(true);
                    if (getContext() != null) {
                        Toast.makeText(getContext(), "状态判断模型文件不存在或已损坏", Toast.LENGTH_SHORT).show();
                    }
                });
            }
            return;
        }

        isModelLoading = true;
        
        getActivity().runOnUiThread(() -> {
            modelLoadingProgress.setVisibility(View.VISIBLE);
            modelStatusText.setText("正在加载聊天模型...");
            loadModelButton.setEnabled(false);
        });
        
        // 在后台线程中加载模型
        new Thread(() -> {
            try {
                Log.d(TAG, "开始加载聊天模型: " + modelPath);
                // 加载聊天模型 (监听器将处理成功加载后的UI更新)
                profileLlamaApi.loadModel(modelPath);
                
                // 聊天模型加载成功后，更新UI并加载状态判断模型
                if (isAdded() && getActivity() != null) {
                    getActivity().runOnUiThread(() -> {
                        modelStatusText.setText("正在加载状态判断模型...");
                    });
                }
                
                // 加载状态判断模型
                Log.d(TAG, "开始加载状态判断模型: " + statusModelFilePath);
                statusLlamaApi.loadModel(statusModelFilePath);
                Log.d(TAG, "状态判断模型加载完成");
                
                // 更新UI，表示所有模型都已准备就绪
                new Handler(Looper.getMainLooper()).post(() -> {
                    if (isAdded() && getActivity() != null) {
                        Toast.makeText(getContext(), "所有模型已准备就绪", Toast.LENGTH_SHORT).show();
                    }
                });
                
            } catch (Exception e) {
                Log.e(TAG, "模型加载失败", e);
                // 主线程更新UI
                new Handler(Looper.getMainLooper()).post(() -> {
                    if (isAdded() && getActivity() != null) {
                        isModelLoading = false;
                        modelLoadingProgress.setVisibility(View.GONE);
                        modelStatusText.setText("模型加载失败: " + e.getMessage());
                        loadModelButton.setEnabled(true);
                        if (getContext() != null) {
                            Toast.makeText(getContext(), "模型加载失败: " + e.getMessage(), Toast.LENGTH_SHORT).show();
                        }
                    }
                });
            }
        }).start();
    }
    
    private void unloadModel() {
        modelStatusText.setText("正在卸载模型...");
        loadModelButton.setEnabled(false);
        
        // 由于使用静态变量共享模型实例，我们不再真正卸载模型
        // 而是仅在应用退出时或内存不足时才考虑卸载
        
        // 仅移除监听器，不卸载模型
        if (profileLlamaApi != null) {
            profileLlamaApi.removeModelStateListener(this);
            Log.d(TAG, "已移除聊天模型监听器，但保留模型实例以便重用");
        }
        
        if (statusLlamaApi != null) {
            statusLlamaApi.removeModelStateListener(this);
            Log.d(TAG, "已移除状态判断模型监听器，但保留模型实例以便重用");
        }
        
        // 更新UI状态
        updateModelStatus();
    }
    
    // 添加静态变量，用于跨Fragment共享模型实例
    private static LLamaAPI sharedProfileLlamaApi;
    private static LLamaAPI sharedStatusLlamaApi;
    
    private void updateModelStatus() {
        try {
            // 检查聊天模型是否已加载
            boolean isModelLoaded = profileLlamaApi != null && profileLlamaApi.isModelLoaded();
            // 检查状态判断模型是否已加载
            boolean isStatusModelLoaded = statusLlamaApi != null && statusLlamaApi.isModelLoaded();
            String currentModelName = profileLlamaApi != null ? profileLlamaApi.getCurrentModelName() : null;
            String currentStatusModelName = statusLlamaApi != null ? statusLlamaApi.getCurrentModelName() : null;
            
            // 只有两个模型都加载完成，才认为模型已加载
            isModelLoaded = isModelLoaded && isStatusModelLoaded;
            
            Log.d(TAG, "updateModelStatus: isModelLoaded = " + isModelLoaded + 
                  ", isModelLoading = " + isModelLoading);
            
            // 检查模型文件是否有效
            String modelFilePath = getModelFilePath();
            File modelFile = new File(modelFilePath);
            boolean modelFileValid = modelFile.exists() && modelFile.length() > 10 * 1024 * 1024; // 确保文件至少有10MB
            
            modelLoadingProgress.setVisibility(isModelLoading && !isModelLoaded ? View.VISIBLE : View.GONE);
            loadModelButton.setEnabled(!isModelLoaded && !isModelLoading);
            
            if (isModelLoaded) {
                if (currentModelName != null && currentStatusModelName != null) {
                    modelStatusText.setText("聊天模型已加载: " + currentModelName + "\n状态判断模型已加载: " + currentStatusModelName);
                } else {
                    modelStatusText.setText("所有模型已加载，可以开始聊天");
                }
            } else if (isModelLoading) {
                modelStatusText.setText("模型加载中...");
                modelLoadingProgress.setVisibility(View.VISIBLE);
            } else {
                // 既不是加载中也不是已加载
                if (modelFileValid) {
                    modelStatusText.setText("模型已准备好但未加载，点击加载按钮开始加载");
                } else {
                    modelStatusText.setText("模型文件未准备好，点击加载按钮准备模型");
                }
            }
        } catch (Exception e) {
            Log.e(TAG, "更新模型状态失败", e);
            modelStatusText.setText("无法获取模型状态");
        }
    }

    @Override
    public void onResume() {
        super.onResume();
        
        // 注册监听器
        if (profileLlamaApi != null) {
            profileLlamaApi.addModelStateListener(this);
        }
        
        // 为状态判断模型注册监听器
        if (statusLlamaApi != null) {
            statusLlamaApi.addModelStateListener(this);
        }
        
        // 更新状态
        updateModelStatus();
        
        // 刷新心理状态数据，确保显示最新的评估结果
        loadPsychologicalStatus();
    }
    
    @Override
    public void onPause() {
        super.onPause();
        
        // 移除监听器
        if (profileLlamaApi != null) {
            profileLlamaApi.removeModelStateListener(this);
        }
        
        // 移除状态判断模型的监听器
        if (statusLlamaApi != null) {
            statusLlamaApi.removeModelStateListener(this);
        }
    }
    
    // 实现ModelStateListener接口
    @Override
    public void onModelLoaded() {
        // 在主线程更新UI
        if (isAdded() && getActivity() != null) {
            Log.d(TAG, "onModelLoaded callback received");
            getActivity().runOnUiThread(() -> {
                isModelLoading = false;
                updateModelStatus();
                if (getContext() != null) {
                    Toast.makeText(getContext(), "模型已加载", Toast.LENGTH_SHORT).show();
                }
            });
        }
    }
    
    @Override
    public void onModelUnloaded() {
        // 在主线程更新UI
        if (isAdded() && getActivity() != null) {
            Log.d(TAG, "onModelUnloaded callback received");
            getActivity().runOnUiThread(() -> {
                // 由于我们使用静态变量共享模型实例，这里只更新UI状态
                // 但不显示Toast，避免在页面跳转时显示"模型已卸载"的提示
                updateModelStatus();
                
                // 检查是否是用户主动卸载模型，只有在这种情况下才显示Toast
                // 页面跳转导致的监听器触发不应显示Toast
                if (getContext() != null && !isStaticInstancesInitialized()) {
                    Toast.makeText(getContext(), "模型已卸载", Toast.LENGTH_SHORT).show();
                }
            });
        }
    }
    
    // 检查静态实例是否已初始化
    private boolean isStaticInstancesInitialized() {
        return sharedProfileLlamaApi != null && sharedStatusLlamaApi != null;
    }

    // 不再需要initDownloadReceiver和checkPreviousDownloadState方法，因为我们从assets加载模型
}