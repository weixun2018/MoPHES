package com.example.project.db;

import android.content.ContentValues;
import android.content.Context;
import android.database.Cursor;
import android.database.sqlite.SQLiteDatabase;
import android.database.sqlite.SQLiteOpenHelper;

import com.example.project.model.Guide;

import java.util.ArrayList;
import java.util.List;

public class GuideDbHelper extends SQLiteOpenHelper {
    private static final String DATABASE_NAME = "guide.db";
    private static final int DATABASE_VERSION = 1;

    private static final String TABLE_GUIDES = "guides";
    private static final String COLUMN_ID = "id";
    private static final String COLUMN_CONTENT = "content";
    private static final String COLUMN_IS_DEFAULT = "is_default";
    private static final String COLUMN_IS_COMPLETED = "is_completed";
    private static final String COLUMN_USER_ID = "user_id";
    private static final String COLUMN_TIMESTAMP = "timestamp";

    public GuideDbHelper(Context context) {
        super(context, DATABASE_NAME, null, DATABASE_VERSION);
    }

    @Override
    public void onCreate(SQLiteDatabase db) {
        String createTable = "CREATE TABLE " + TABLE_GUIDES + " (" +
                COLUMN_ID + " INTEGER PRIMARY KEY AUTOINCREMENT, " +
                COLUMN_CONTENT + " TEXT, " +
                COLUMN_IS_DEFAULT + " INTEGER, " +
                COLUMN_IS_COMPLETED + " INTEGER, " +
                COLUMN_USER_ID + " INTEGER, " +
                COLUMN_TIMESTAMP + " INTEGER)";
        db.execSQL(createTable);
        
        // 插入默认指南
        insertDefaultGuides(db);
    }

    @Override
    public void onUpgrade(SQLiteDatabase db, int oldVersion, int newVersion) {
        db.execSQL("DROP TABLE IF EXISTS " + TABLE_GUIDES);
        onCreate(db);
    }
    
    /**
     * 插入默认的5条治疗方案
     */
    private void insertDefaultGuides(SQLiteDatabase db) {
        String[] defaultGuides = {
            "每天进行30分钟有氧运动，如散步、慢跑或游泳，有助于减轻焦虑和抑郁症状",
            "学习并练习深呼吸和冥想技巧，每天花10-15分钟进行正念练习",
            "保持规律的作息时间，确保每晚7-8小时的充足睡眠",
            "与亲友保持联系，分享你的感受，不要孤立自己",
            "限制咖啡因和酒精的摄入，这些物质可能加重焦虑和抑郁症状"
        };
        
        for (String content : defaultGuides) {
            ContentValues values = new ContentValues();
            values.put(COLUMN_CONTENT, content);
            values.put(COLUMN_IS_DEFAULT, 1); // 1表示默认治疗方案
            values.put(COLUMN_IS_COMPLETED, 0); // 0表示未完成
            values.put(COLUMN_USER_ID, 0); // 0表示系统默认治疗方案
            values.put(COLUMN_TIMESTAMP, System.currentTimeMillis());
            db.insert(TABLE_GUIDES, null, values);
        }
    }

    /**
     * 插入新治疗方案
     */
    public long insertGuide(Guide guide) {
        SQLiteDatabase db = this.getWritableDatabase();
        ContentValues values = new ContentValues();
        values.put(COLUMN_CONTENT, guide.getContent());
        values.put(COLUMN_IS_DEFAULT, guide.isDefault() ? 1 : 0);
        values.put(COLUMN_IS_COMPLETED, guide.isCompleted() ? 1 : 0);
        values.put(COLUMN_USER_ID, guide.getUserId());
        values.put(COLUMN_TIMESTAMP, guide.getTimestamp());
        long id = db.insert(TABLE_GUIDES, null, values);
        guide.setId(id);
        return id;
    }

    /**
     * 更新治疗方案完成状态
     */
    public boolean updateGuideCompletionStatus(long guideId, boolean isCompleted) {
        SQLiteDatabase db = this.getWritableDatabase();
        ContentValues values = new ContentValues();
        values.put(COLUMN_IS_COMPLETED, isCompleted ? 1 : 0);
        
        int rowsAffected = db.update(
            TABLE_GUIDES, 
            values, 
            COLUMN_ID + " = ?", 
            new String[] { String.valueOf(guideId) }
        );
        
        return rowsAffected > 0;
    }

    /**
     * 获取用户的所有治疗方案（包括系统默认治疗方案和用户自定义治疗方案）
     */
    public List<Guide> getGuidesForUser(long userId) {
        List<Guide> guides = new ArrayList<>();
        SQLiteDatabase db = this.getReadableDatabase();
        
        // 查询条件：系统默认指南(user_id=0)或当前用户的自定义指南
        String selectQuery = "SELECT * FROM " + TABLE_GUIDES + 
                " WHERE " + COLUMN_USER_ID + " = 0 OR " + 
                COLUMN_USER_ID + " = ? ORDER BY " + 
                COLUMN_IS_DEFAULT + " DESC, " + 
                COLUMN_TIMESTAMP + " ASC";
                
        Cursor cursor = db.rawQuery(selectQuery, new String[]{String.valueOf(userId)});

        if (cursor.moveToFirst()) {
            do {
                Guide guide = new Guide();
                guide.setId(cursor.getLong(cursor.getColumnIndex(COLUMN_ID)));
                guide.setContent(cursor.getString(cursor.getColumnIndex(COLUMN_CONTENT)));
                guide.setDefault(cursor.getInt(cursor.getColumnIndex(COLUMN_IS_DEFAULT)) == 1);
                guide.setCompleted(cursor.getInt(cursor.getColumnIndex(COLUMN_IS_COMPLETED)) == 1);
                guide.setUserId(cursor.getLong(cursor.getColumnIndex(COLUMN_USER_ID)));
                guide.setTimestamp(cursor.getLong(cursor.getColumnIndex(COLUMN_TIMESTAMP)));
                guides.add(guide);
            } while (cursor.moveToNext());
        }
        cursor.close();
        return guides;
    }
    
    /**
     * 删除用户自定义指南
     */
    public boolean deleteUserGuide(long guideId, long userId) {
        SQLiteDatabase db = this.getWritableDatabase();
        // 只能删除自己的自定义指南，不能删除系统默认指南
        return db.delete(TABLE_GUIDES, 
                COLUMN_ID + " = ? AND " + 
                COLUMN_USER_ID + " = ? AND " + 
                COLUMN_IS_DEFAULT + " = 0", 
                new String[] { String.valueOf(guideId), String.valueOf(userId) }) > 0;
    }
}