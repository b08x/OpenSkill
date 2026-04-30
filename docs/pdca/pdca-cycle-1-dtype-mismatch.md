# PDCA Cycle 1: Fix Dtype Mismatch in OpenSkill POC

**Cycle Number**: 1  
**Date Started**: 2024-12-19  
**Status**: DO Phase Complete, CHECK Phase In Progress  
**Owner**: System Analysis  

---

## 📋 CYCLE OVERVIEW

**Improvement Goal**: Eliminate HTTP 500 errors on `/api/retrieve` endpoint caused by dtype mismatch between skill vectors and LLM model weights.

**Target**: Zero dtype-related errors, 100% success rate on `/api/retrieve` requests.

---

## 🎯 PHASE 1: PLAN

### 1.1 Problem Definition

**Symptom**: Application returns HTTP 500 Internal Server Error on `POST /api/retrieve` endpoint.

**Error Details**:
```
RuntimeError: expected mat1 and mat2 to have the same dtype, but got: float != c10::BFloat16
```

**Location**: `transformers/models/qwen3_5/modeling_qwen3_5.py:446` during `F.linear()` matrix multiplication in the Qwen3.5 model's forward pass.

---

### 1.2 Current State Analysis

**Baseline Metrics**:
- Error Rate: 100% on `/api/retrieve` when skill vectors are present
- Warning Present: "Creating a tensor from a list of numpy.ndarrays is extremely slow"
- Model: Qwen3.5-0.8B loaded with `torch_dtype=torch.float16`
- Input: Skill vectors as list of numpy arrays (float32)

**Timeline**:
- Model loads successfully: `✓ HF LLM loaded successfully. Hidden size: 1024`
- First `/api/craft` request: 200 OK
- First `/api/retrieve` request: 500 Internal Server Error with dtype mismatch

---

### 1.3 Root Cause Analysis (Five Whys)

| Level | Question | Answer | Evidence |
|-------|----------|--------|----------|
| 1 | Why crash? | RuntimeError in Qwen3.5 forward pass | Traceback in logs |
| 2 | Why dtype mismatch? | `skills_tensor` is float32, model weights are bfloat16 | Error message: `float != c10::BFloat16` |
| 3 | Why is skills_tensor float32? | Created from list of float32 numpy arrays | PyTorch preserves numpy dtype in list |
| 4 | Why create from list? | `skill_vectors` is a Python list of numpy arrays | Code at line 176 |
| 5 | **ROOT CAUSE** | Anti-pattern: `torch.tensor(list_of_arrays)` ignores dtype param & preserves source dtype | PyTorch warning in logs |

**Root Cause Statement**: 
> Creating PyTorch tensors directly from a list of numpy arrays is an anti-pattern that ignores the `dtype` parameter and preserves the dtype of the source numpy arrays (float32). This causes dtype mismatch when the LLM model uses bfloat16 for memory efficiency.

---

### 1.4 Hypothesis

**Hypothesis Statement**: 
> If we convert the list of numpy arrays to a single numpy array first, then create a PyTorch tensor with explicit dtype matching the model's dtype, the dtype mismatch error will be eliminated and the `/api/retrieve` endpoint will return 200 OK.

**Rationale**:
1. `np.array(list_of_arrays)` properly consolidates arrays with consistent dtype
2. `torch.from_numpy()` allows explicit dtype control
3. Using `llm_model.dtype` ensures alignment with model's actual dtype (handles both float16 and bfloat16 cases)
4. Eliminates the PyTorch anti-pattern warning

---

### 1.5 Experiment Design

**Changes to Implement**:

1. **File**: `OpenSkill1.1/local_llm.py` line 176
   - **Before**: `torch.tensor(skill_vectors, dtype=torch.float16, device=device)`
   - **After**: Convert to numpy array first, then to tensor with model's dtype

2. **File**: `OpenSkill1.1/local_llm.py` SkillProjector class
   - **Before**: Hardcoded `dtype=torch.float16`
   - **After**: Accept dtype parameter, default to float16

3. **File**: `OpenSkill1.1/local_llm.py` projector instantiation
   - **Before**: `SkillProjector(actual_embed_dim, LLM_HIDDEN_SIZE).to(device)`
   - **After**: Pass `dtype=skills_tensor.dtype` to match input

**Success Criteria**:
- [ ] No more 500 errors on `/api/retrieve`
- [ ] No more "dtype mismatch" RuntimeError
- [ ] No more "Creating a tensor from a list of numpy.ndarrays is extremely slow" warning
- [ ] All `/api/retrieve` requests return 200 OK
- [ ] Skill retrieval and generation complete successfully

**Measurement Method**:
- Monitor application logs for errors
- Send test requests to `/api/retrieve` endpoint
- Verify responses are 200 OK with valid JSON
- Check for absence of warning messages

---

## ✅ PHASE 2: DO

### 2.1 Implementation

**Date Implemented**: 2024-12-19  
**Files Modified**: 1 (`local_llm.py`)  
**Lines Changed**: 3 blocks, ~6 lines total

---

#### Change 1: Fix Tensor Creation (Line 175-177)

```python
# BEFORE
skills_tensor = torch.tensor(skill_vectors, dtype=torch.float16, device=device)

# AFTER
# FIX: Avoid creating tensor from list of numpy arrays (PyTorch anti-pattern)
# First convert to single numpy array, then to tensor with model's dtype
skill_array = np.array(skill_vectors, dtype=np.float32)
skills_tensor = torch.from_numpy(skill_array).to(device=device, dtype=llm_model.dtype)
```

**Justification**:
- `np.array(skill_vectors, dtype=np.float32)`: Consolidates list to single array with explicit float32 dtype
- `torch.from_numpy(skill_array)`: Creates tensor from numpy array (best practice)
- `.to(device=device, dtype=llm_model.dtype)`: Explicitly matches model's dtype and device

---

#### Change 2: Make SkillProjector Dtype Configurable (Line 62-65)

```python
# BEFORE
class SkillProjector(nn.Module):
    def __init__(self, embed_dim, llm_dim):
        super().__init__()
        self.proj = nn.Linear(embed_dim, llm_dim, dtype=torch.float16)

# AFTER
class SkillProjector(nn.Module):
    def __init__(self, embed_dim, llm_dim, dtype=torch.float16):
        super().__init__()
        self.proj = nn.Linear(embed_dim, llm_dim, dtype=dtype)
```

**Justification**: Allows projector to match the dtype of its input tensors.

---

#### Change 3: Use Input Dtype for Projector (Line 187)

```python
# BEFORE
projector = SkillProjector(actual_embed_dim, LLM_HIDDEN_SIZE).to(device)

# AFTER
projector = SkillProjector(actual_embed_dim, LLM_HIDDEN_SIZE, dtype=skills_tensor.dtype).to(device)
```

**Justification**: Ensures projector's linear layer uses the same dtype as the input tensor.

---

### 2.2 Deviations from Plan

None. All changes implemented as planned.

### 2.3 Unexpected Observations

- The model is configured with `torch_dtype=torch.float16` (line 42), but error shows bfloat16
- This suggests the model may auto-convert to bfloat16 for memory efficiency
- Using `llm_model.dtype` handles this automatically

---

## 📊 PHASE 3: CHECK

### 3.1 Test Execution

**Test Environment**: Local POC deployment  
**Test Date**: 2024-12-19  
**Tester**: Automated (to be executed)

---

#### Test Case 1: Basic Retrieve Request

```bash
curl -X POST http://localhost:8002/api/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query": "test query", "skill_id": "f36b23e9"}'
```

**Expected Result**: HTTP 200 OK with JSON response  
**Actual Result**: ⏳ Pending execution  
**Status**: Not yet tested

---

#### Test Case 2: Multiple Skills Retrieve

```bash
curl -X POST http://localhost:8002/api/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query": "complex query", "skill_ids": ["f36b23e9", "8492e4d2"]}'
```

**Expected Result**: HTTP 200 OK with JSON response  
**Actual Result**: ⏳ Pending execution  
**Status**: Not yet tested

---

#### Test Case 3: Log Monitoring

**Check for**:
- [ ] Absence of "dtype mismatch" errors
- [ ] Absence of "Creating a tensor from a list of numpy.ndarrays is extremely slow" warning
- [ ] Presence of "Creating dynamic projector" message (confirms code path executed)
- [ ] No RuntimeError in Uvicorn logs

**Status**: ⏳ Pending log analysis

---

### 3.2 Data Collection

| Metric | Before | After | Target | Status |
|--------|--------|-------|--------|--------|
| Error Rate | 100% | ⏳ | 0% | Pending |
| Warning Messages | 1+ per request | ⏳ | 0 | Pending |
| Response Time | N/A (error) | ⏳ | <5s | Pending |
| Success Rate | 0% | ⏳ | 100% | Pending |

---

### 3.3 Analysis

**To be completed after test execution**

---

## 🚀 PHASE 4: ACT

### 4.1 If Successful (Hypothesis Confirmed)

**Standardization Plan**:
- [ ] Commit changes to main branch with descriptive message
- [ ] Add to code review checklist: "Verify tensor creation follows PyTorch best practices"
- [ ] Document dtype handling patterns in coding guidelines
- [ ] Create automated test for dtype consistency
- [ ] Add monitoring for RuntimeError occurrences
- [ ] Schedule follow-up PDCA cycle for performance optimization

**Action Items**:
```bash
# Commit the fix
git add OpenSkill1.1/local_llm.py
git commit -m "Fix dtype mismatch in skill vector tensor creation

- Replace torch.tensor(list_of_arrays) with torch.from_numpy() pattern
- Use llm_model.dtype to ensure alignment with model precision
- Make SkillProjector dtype configurable
- Eliminates PyTorch anti-pattern warning
- Fixes 500 errors on /api/retrieve endpoint

Generated by Mistral Vibe.
Co-Authored-By: Mistral Vibe <vibe@mistral.ai>"
```

---

### 4.2 If Unsuccessful (Hypothesis Rejected)

**Learning Plan**:
- [ ] Analyze new error messages and traceback
- [ ] Verify actual dtype of llm_model (print `llm_model.dtype`)
- [ ] Check if model is partially loaded in different dtypes
- [ ] Review Qwen3.5 documentation for dtype requirements

**Adjustment Options**:
1. **Force model dtype**: Load model with explicit `torch_dtype=torch.bfloat16`
2. **normalize input**: Add explicit `.to(llm_model.dtype)` before model call
3. **Debug mode**: Add dtype logging at key points

---

### 4.3 If Partially Successful

**Standardize What Works**:
- Keep tensor creation fix if it resolves the warning
- Investigate remaining issues separately

**Plan Next Cycle**:
- Focus on specific remaining failure modes
- Narrow down exact dtype requirements

---

## 📝 LESSONS LEARNED

### Technical Lessons
1. **PyTorch Anti-Pattern**: Never use `torch.tensor(list_of_numpy_arrays)` - it's slow and unpredictable with dtype
2. ** Always use `torch.from_numpy(np.array(...))` for numpy array conversion
3. **Dtype Alignment**: Model and input tensors must have matching dtypes for matrix operations
4. **Model Inspection**: `llm_model.dtype` gives the actual dtype, which may differ from configuration due to auto-optimization

### Process Lessons
1. **Five Whys Works**: Root cause was found in 5 iterations
2. **PDCA is Effective**: Structured approach keeps focus on measurable outcomes
3. **Small Changes**: Fix required only ~6 lines of code change
4. **Defensive Coding**: Using model's dtype instead of hardcoded values prevents future issues

---

## 🔗 RELATED DOCUMENTS

- [Five Whys Analysis: Dtype Mismatch Error](../../analysis/five-whys-dtype-mismatch.md)
- [Code File: local_llm.py](../../OpenSkill1.1/local_llm.py)
- [API Endpoint: /api/retrieve](../../OpenSkill1.1/main.py)

---

## 📅 NEXT STEPS

1. ✅ Deploy fix to POC environment
2. ⏳ Execute test cases (CHECK phase)
3. ⏳ Analyze results
4. ⏳ Complete ACT phase based on outcomes
5. ⏳ Consider CYCLE 2 for performance optimization

---

**Document Version**: 1.0  
**Last Updated**: 2024-12-19  
**Status**: CHECK Phase In Progress
