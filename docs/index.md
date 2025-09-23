---
layout: default
title: "codeit AI 4기 4팀 초급 프로젝트"
description: "codeit AI 4기 4팀 초급 프로젝트 "
date: 2025-09-09
author: "김명환"
cache-control: no-cache
expires: 0
pragma: no-cache
---

## 🏥 코드잇 AI 엔지니어 4기 4팀 초급 프로젝트

### 📱 프로젝트 개요
**제목**: 모바일 애플리케이션으로 촬영한 약물 이미지에서 **최대 4개의 알약을 동시에 검출하고 분류**하여, 사용자에게 약물 정보 및 상호작용 경고를 제공하는 AI 시스템 개발

### 👥 팀원

| 역할 | 담당자 | 핵심 업무 |
|------|--------|-----------|
| **Project Manager** | [이건희](https://github.com/Lee-keonhee) | 프로젝트 총괄 관리, 일정 조율 |
| **Data Engineer** | 서동일 | EDA, 데이터 전처리, 증강 기법 |
| **Model Architect** | [김명환](https://c0z0c.github.io/) | YOLO v8 + EfficientNet-B3 설계 |
| **Experimentation Lead** | [김민혁](https://github.com/cmustard7) | 실험 설계, Kaggle 제출, 성능 튜닝 |
| **Quality Assurance** | 이현재 | 코드 품질, 문서화, 결과 검증 |


### 📅 프로젝트 기간
**2025년 9월 9일 ~ 2025년 9월 25일**

<hr style="margin: 30px 0;">

<script>

{% assign cur_dir = "/" %}
{% include cur_files.liquid %}

  var curDir = '{{- cur_file_dir -}}';
  var curFiles = {{- cur_files_json -}};

  var curPages = allPages.filter(page => {
    return page.dir === '/' || page.dir.startsWith('/md/');
  });  
  
  console.log('curDir:', curDir);
  console.log('curFiles:', curFiles);
  console.log('curPages:', curPages);

  curPages.forEach(page => {
    // curFiles에 같은 name과 path가 있는지 확인
    const exists = curFiles.some(file => file.name === page.name && file.path === page.path);

    if (!exists) {
      // 확장자 추출
      let extname = '';
      if (page.name && page.name.includes('.')) {
        extname = '.' + page.name.split('.').pop();
      }

      // basename 추출
      let basename = page.name ? page.name.replace(/\.[^/.]+$/, '') : '';

      // modified_time 처리 (page.date가 없으면 빈 문자열)
      let modified_time = page.date || '';

      // curFiles 포맷에 맞게 변환해서 추가
      curFiles.push({
        name: page.name || '',
        path: page.path || '',
        extname: extname,
        modified_time: modified_time,
        basename: basename,
        url: page.url || ''
      });
    }
  });


  curFiles.sort((a, b) => {
    // 파일명으로 한글/영문 구분하여 정렬
    if (!a.name) return 1;
    if (!b.name) return -1;
    return a.name.localeCompare(b.name, 'ko-KR', { numeric: true, caseFirst: 'lower' });
  });

  console.log('총 파일 수:', curFiles.length);
  console.log('파일 목록:', curFiles);

  var project_path = site.baseurl
  var project_url = `https://c0z0c.github.io${project_path}`
  var project_git_url = `https://github.com/c0z0c${project_path}/blob/${branch}/`
  var site_url = `https://c0z0c.github.io${project_path}${curDir}`
  var raw_url = `https://raw.githubusercontent.com/c0z0c${project_path}/alpha${curDir}`;
  var git_url = `https://github.com/c0z0c${project_path}/blob/${branch}/docs${curDir}`
  var colab_url = `https://colab.research.google.com/github/c0z0c${project_path}/blob/alpha${curDir}`;

  console.log('project_url:', project_url);
  console.log('project_git_url:', project_git_url);
  console.log('site_url:', site_url);
  console.log('raw_url:', raw_url);
  console.log('git_url:', raw_url);
  console.log('colab_url:', colab_url);


// 파일 아이콘 및 타입 결정 함수
  function getFileInfo(extname) {
    switch(extname.toLowerCase()) {
      case '.md':
        return { icon: '📝', type: 'Markdown 문서' };
      default:
        return { icon: '📄', type: '파일' };
    }
  }

// 파일 액션 버튼 생성 함수
  function getFileActions(file) {
    const fileName = file.name;
    const fileExt = file.extname.toLowerCase();
    const githubRawUrl = `${raw_url}${fileName}`;
    
    let actions = '';
    
    // Markdown 파일 처리
    if (fileExt === '.md' && fileName !== 'index.md') {
      const mdName = fileName.replace('.md', '');
      actions += `<a href="${site_url}md/${mdName}" class="file-action" title="렌더링된 페이지 보기">🌐</a>`;
      actions += `<a href="${git_url}md/${fileName}" class="file-action" title="GitHub에서 원본 보기" target="_blank">📖</a>`;
    } 
    // 기타 파일
    else {
      actions += `<a href="${git_url}${fileName}" class="file-action" title="GitHub에서 원본 보기" target="_blank">📖</a>`;
    }
    return actions;
  }

  // DOM이 로드된 후 파일 목록 렌더링
  document.addEventListener('DOMContentLoaded', function() {
    const fileGrid = document.querySelector('.file-grid');
    
    if (curFiles.length === 0) {
      fileGrid.innerHTML = `
        <div class="empty-message">
          <span class="empty-icon">📄</span>
          <h3>파일이 없습니다</h3>
          <p>현재 이 위치에는 완료된 미션 파일이 없습니다.</p>
        </div>
      `;
      return;
    }

    let html = '';
    curFiles.forEach(file => {
      if (file.name === 'index.md' || file.name === 'info.md') return;

      const fileInfo = getFileInfo(file.extname);
      const fileDate = file.modified_time ? new Date(file.modified_time).toLocaleDateString('ko-KR') : '';
      const actions = getFileActions(file);
      
      html += `
        <div class="file-item">
          <div class="file-icon">${fileInfo.icon}</div>
          <div class="file-info">
            <h4 class="file-name">${file.name}</h4>
            <p class="file-type">${fileInfo.type}</p>
            <p class="file-size">${fileDate}</p>
          </div>
          <div class="file-actions">
            ${actions}
          </div>
        </div>
      `;
    });
    
    fileGrid.innerHTML = html;
  });

{% include page_folders.html %}

</script>

<h2>�📖 프로젝트 문서 목록</h2>
<div class="file-grid">
  <!-- 파일 목록이 JavaScript로 동적 생성됩니다 -->
</div>

<h2>� 하위 폴더 목록</h2>
<div class="folder-grid">
  <!-- 폴더 목록이 JavaScript로 동적 생성됩니다 -->
</div>



