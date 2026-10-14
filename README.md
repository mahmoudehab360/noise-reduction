# LowLight DN

برنامج إزالة ضوضاء الإضاءة الضعيفة من الفيديو (Temporal NR + Spatial NR) على كرت NVIDIA، مكتوب بـ Rust + wgpu (Vulkan / DX12).

## البناء على GitHub
1. ارفع كل ملفات المشروع للـ repo (مع مجلد `.github/workflows`).
2. من تبويب Actions شغّل `build-windows` (أو يشتغل تلقائي مع أي push).
3. بعد ما يخلص نزّل الـ artifact باسم `LowLight-DN-windows` وفك الضغط.
4. شغّل `lowlight-dn.exe` (لازم `ffmpeg.exe` و `ffprobe.exe` يفضلوا جنبه في نفس المجلد).

## الاستخدام
- Open video أو اسحب الملف على النافذة.
- Auto-detect noise level لضبط الـ thresholds تلقائيًا، وبعدها عدّل بإيدك.
- المعاينة بتتحدث لأي فريم على التايملاين وأي تغيير في الإعدادات.
- حدد In/Out (مفتاحي I و O) واضغط Play لمعاينة جزء من الفيديو بالفلاتر.
- Export يحفظ المدى المحدد مع الصوت (H.264 / H.265 / ProRes).

## اختصارات
Space = تشغيل/إيقاف، الأسهم = فريم بفريم، I / O = بداية ونهاية المدى.

## نصائح للأداء
- أثناء التشغيل: Spatial Mode = Faster و Radius = Small.
- للتصدير النهائي: Better أو Enhanced.
- Frames (each side) من 2 لـ 3 يعطي أقوى نتيجة على اللقطات الثابتة.
