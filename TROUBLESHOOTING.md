# 🆘 TROUBLESHOOTING GUIDE

## ✅ Current Status Check

### Is the Server Running?
Open PowerShell in the project folder and check:
```powershell
cd "C:\Users\Rudra\OneDrive\Desktop\Manthan"
python app.py
```

**Look for these lines:**
```
[Model 2.0-NLP-Enhanced] Initialized with 500+ transformation rules across 4 datasets
🚀 Starting Tone Converter Application
📍 Server running at: http://localhost:5000
📊 Model Status: Loaded
* Running on http://127.0.0.1:5000
* Debugger is active!
```

If you see these, **SERVER IS RUNNING!** ✅

---

## 🔧 Quick Fixes

### Option 1: Open the Simple Test File
**Double-click:** `SIMPLE_TEST.html` in your project folder

This will:
- Open in your browser
- Show a big green button "TEST CONVERSION"
- Click it to test if API is working
- Shows clear error messages if something is wrong

### Option 2: Open Main Application
**In browser, go to:** http://localhost:5000

OR click this file: `test_standalone.html`

### Option 3: Test from Command Line
```powershell
.\quick_test.ps1
```

This will test the API and show results immediately.

---

## 🐛 Common Issues & Solutions

### Problem 1: "Cannot connect to server"
**Solution:**
```powershell
# Stop any running Python
Get-Process python -ErrorAction SilentlyContinue | Stop-Process

# Restart server
python app.py
```

### Problem 2: "Server starts but browser shows nothing"
**Solution:**
1. Close ALL browser tabs with localhost:5000
2. Clear browser cache (Ctrl+Shift+Delete)
3. Try different browser (Edge, Chrome, Firefox)
4. Try: http://127.0.0.1:5000 instead

### Problem 3: "Port 5000 already in use"
**Solution:**
```powershell
# Find what's using port 5000
netstat -ano | findstr :5000

# Kill that process (replace PID with the number you see)
taskkill /PID [PID] /F

# Restart server
python app.py
```

### Problem 4: "Module not found errors"
**Solution:**
```powershell
# Reinstall dependencies
python -m pip install --upgrade pip
python -m pip install flask flask-cors
```

### Problem 5: "Conversions not working / No alternatives"
**Solution:**
Server needs to reload after code changes:
```powershell
# In terminal running server, press Ctrl+C to stop
# Then restart:
python app.py
```

The debug mode should auto-reload, but sometimes it needs manual restart.

---

## 🧪 Testing Tools Included

### 1. SIMPLE_TEST.html ⭐ RECOMMENDED
- **How:** Double-click the file
- **What:** Simple interface to test API
- **Shows:** Clear errors and success messages

### 2. test_standalone.html
- **How:** Double-click the file  
- **What:** Full standalone version
- **Shows:** Complete UI with all features

### 3. test_model.py
- **How:** `python test_model.py`
- **What:** Tests the AI model directly
- **Shows:** Conversions and alternatives for 4 test cases

### 4. quick_test.ps1
- **How:** `.\quick_test.ps1`
- **What:** PowerShell API test
- **Shows:** API response with details

---

## 📊 Verify Everything is Working

### Step 1: Test the Model Directly
```powershell
python test_model.py
```

**Expected Output:**
```
✅ Polite mode: 2 alternatives
✅ Formal mode: 2 alternatives  
✅ Professional mode: 2 alternatives
✅ Friendly mode: 2 alternatives
```

If you see this, **MODEL IS WORKING!** ✅

### Step 2: Test the API
```powershell
.\quick_test.ps1
```

**Expected Output:**
```
✅ SUCCESS!
Original: I need help with this task
Converted: I would appreciate if I could have assistance with this task...
Alternatives: 2
```

If you see this, **API IS WORKING!** ✅

### Step 3: Test in Browser
Open: `SIMPLE_TEST.html` and click the button

If you see conversion results, **EVERYTHING IS WORKING!** ✅

---

## 🎯 What Should Be Working

After fixes, you should have:

✅ **Polite Mode**
- Input: "I need this now"
- Output: "I would appreciate if I could have this at your earliest convenience. Thank you for your attention to this."
- Alternatives: 2 suggestions

✅ **Formal Mode**
- Input: "Hey, I can't make it"
- Output: "I cannot make it to the meeting"
- Alternatives: 2 suggestions

✅ **Professional Mode**
- Input: "We need to fix this problem"
- Output: "We require to resolve this issue"
- Alternatives: 2 suggestions

✅ **Friendly Mode**
- Input: "Thanks for helping"
- Output: "Hi! Thanks for helping Best wishes!"
- Alternatives: 2 suggestions

✅ **All Modes**
- Always show 2 alternative suggestions
- Show detailed processing info
- Display confidence scores
- Show AI processing notes

---

## 🔍 Check Server Logs

While server is running, the terminal shows:
- Every HTTP request received
- Any errors that occur
- Model initialization status

**To see logs:**
Look at the PowerShell window where you ran `python app.py`

**Healthy logs look like:**
```
127.0.0.1 - - [21/Nov/2025 10:30:45] "GET / HTTP/1.1" 200 -
127.0.0.1 - - [21/Nov/2025 10:30:50] "POST /api/convert HTTP/1.1" 200 -
```

**Problem logs look like:**
```
127.0.0.1 - - [21/Nov/2025 10:30:50] "POST /api/convert HTTP/1.1" 500 -
Error in convert_text: ...
```

---

## 🆘 Still Not Working?

### Try This Emergency Reset:

```powershell
# 1. Kill everything
Get-Process python -ErrorAction SilentlyContinue | Stop-Process

# 2. Clean install
python -m pip uninstall flask flask-cors -y
python -m pip install flask flask-cors

# 3. Test model directly (should work even if server doesn't)
python test_model.py

# 4. Restart server
python app.py

# 5. Open simple test
Start-Process "SIMPLE_TEST.html"
```

---

## 📞 What to Report if Still Broken

If still not working, provide:

1. **Server output**: Copy-paste from terminal where `python app.py` runs
2. **Test results**: Output from `python test_model.py`
3. **Browser console**: Press F12 in browser, go to Console tab, screenshot any red errors
4. **What you see**: Screenshot of browser when you open localhost:5000

---

## ✅ Success Checklist

Your app is fully working when:

- [ ] `python test_model.py` shows 4 tests with 2 alternatives each
- [ ] `python app.py` starts server without errors
- [ ] Opening http://localhost:5000 shows the UI
- [ ] Typing text and clicking Convert shows results
- [ ] Results show 2 alternative suggestions
- [ ] Processing details appear below results
- [ ] Different modes produce different outputs

---

**Last Updated:** November 21, 2025  
**Project:** AI Tone Converter v2.0  
**Status:** All components tested and working ✅
