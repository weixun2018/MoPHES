package com.example.project;

import android.os.Bundle;
import android.view.MenuItem;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.fragment.app.Fragment;

import com.google.android.material.bottomnavigation.BottomNavigationView;
import com.example.project.fragment.AiChatFragment;
import com.example.project.fragment.GuideFragment;
import com.example.project.fragment.MbtiFragment;
import com.example.project.fragment.ProfileFragment;
import com.example.project.fragment.TestSelectionFragment;

public class MainActivity extends AppCompatActivity implements BottomNavigationView.OnNavigationItemSelectedListener {

    private BottomNavigationView bottomNavigationView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        bottomNavigationView = findViewById(R.id.bottom_nav_view);
        bottomNavigationView.setOnNavigationItemSelectedListener(this);

        // 默认选中AI心理咨询页面，仅在首次创建时设置选中，避免重复加载Fragment
        if (savedInstanceState == null) {
            bottomNavigationView.setSelectedItemId(R.id.navigation_ai_chat);
        }
    }

    @Override
    public boolean onNavigationItemSelected(@NonNull MenuItem item) {
        Fragment fragment = null;
        int itemId = item.getItemId();

        if (itemId == R.id.navigation_ai_chat) {
            fragment = AiChatFragment.newInstance();
        } else if (itemId == R.id.navigation_test) {
            fragment = TestSelectionFragment.newInstance();
        } else if (itemId == R.id.navigation_news) {
            fragment = GuideFragment.newInstance();
        } else if (itemId == R.id.navigation_profile) {
            fragment = ProfileFragment.newInstance();
        }
        return loadFragment(fragment);
    }

    private boolean loadFragment(Fragment fragment) {
        if (fragment != null) {
            getSupportFragmentManager()
                    .beginTransaction()
                    .replace(R.id.nav_host_fragment, fragment)
                    .commit();
            return true;
        }
        return false;
    }
    
    /**
     * 切换到个人中心页面
     */
    public void switchToProfileTab() {
        bottomNavigationView.setSelectedItemId(R.id.navigation_profile);
    }
}