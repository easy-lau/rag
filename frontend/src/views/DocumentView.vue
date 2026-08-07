<template>
  <div class="p-4 sm:p-6 h-full overflow-y-auto">
    <div class="max-w-6xl mx-auto space-y-5">
      <PageHeader title="文档管理" description="上传、审阅和维护知识库中的文档内容与检索标签。">
        <template #actions>
          <div class="flex flex-wrap items-center gap-2">
            <n-button secondary size="small" aria-label="返回知识库管理" @click="$router.push({ name: 'knowledge' })">
              <template #icon><n-icon><ArrowBackOutline /></n-icon></template>
              <span class="hidden sm:inline">返回知识库</span>
            </n-button>
        <n-select v-model:value="selectedKbId" :options="kbOptions" placeholder="选择知识库" class="w-48" clearable />
        <template v-if="canCreateDocument">
          <n-button :disabled="!selectedKbId" @click="showUpload = true">
            <template #icon><n-icon><CloudUploadOutline /></n-icon></template>
            上传文档
          </n-button>
          <n-button :disabled="!selectedKbId" @click="showImageUpload = true">
            <template #icon><n-icon><ImageOutline /></n-icon></template>
            上传图片
          </n-button>
          <n-button type="primary" :disabled="!selectedKbId" @click="openTextEditor">
            <template #icon><n-icon><CreateOutline /></n-icon></template>
            手动输入
          </n-button>
        </template>
          </div>
        </template>
      </PageHeader>

    <!-- 未选知识库时的友好空状态（正常从知识库卡片进入会带 kb，不会到这里） -->
    <SurfaceCard v-if="!selectedKbId" class="flex flex-col items-center justify-center py-24 text-center">
      <n-icon :size="40" class="mb-3"><LibraryOutline /></n-icon>
      <p class="text-sm text-gray-500 dark:text-gray-400 mb-1">请选择一个知识库来管理其文档</p>
      <p class="text-xs mb-4">文档归属于知识库，可在上方下拉选择，或从知识库管理进入</p>
      <n-button size="small" @click="$router.push({ name: 'knowledge' })">前往知识库管理</n-button>
    </SurfaceCard>

    <template v-else>
      <div v-if="checkedRowKeys.length" class="flex items-center gap-3 mb-3">
        <span class="text-sm text-gray-500 dark:text-gray-400">已选 {{ checkedRowKeys.length }} 项</span>
        <n-button
          v-if="checkedDraftCount && checkedDraftCount === checkedRowKeys.length"
          size="small" type="primary" :loading="batchIngesting" @click="submitBatchIngest"
        >
          <template #icon><n-icon><CloudDoneOutline /></n-icon></template>
          保存入库（{{ checkedDraftCount }}）
        </n-button>
        <n-button v-if="canDeleteDocument" size="small" type="error" @click="openBatchDelete">
          <template #icon><n-icon><TrashOutline /></n-icon></template>
          批量删除
        </n-button>
        <n-button size="small" text @click="checkedRowKeys = []">取消选择</n-button>
      </div>
      <SurfaceCard padding="none" class="overflow-hidden">
        <n-data-table
          :columns="columns" :data="docs" :loading="loading"
          :row-key="rowKey" v-model:checked-row-keys="checkedRowKeys"
          :pagination="pagination" :scroll-x="1420"
          class="admin-data-table"
        />
      </SurfaceCard>
    </template>

    <!-- Upload modal -->
    <AppModal v-model:show="showUpload" title="上传文档" width="min(90vw, 480px)" :loading="uploading">
      <div class="relative">
        <div
          v-if="uploading"
          class="absolute inset-0 z-10 flex flex-col items-center justify-center gap-3 bg-white/85 dark:bg-gray-800/85 backdrop-blur-sm rounded-lg"
        >
          <n-spin size="large" />
          <span class="text-sm text-gray-600 dark:text-gray-300 font-medium">{{ uploadStatus }}</span>
        </div>
        <n-upload
          multiple :custom-request="handleUpload"
          accept=".pdf,.docx,.doc,.pptx,.ppt,.txt,.md,.xlsx,.xls"
          :default-upload="false"
          v-model:file-list="fileList"
          :disabled="uploading"
        >
          <n-upload-dragger>
            <div class="py-6 text-center">
              <n-icon :size="36" class="text-blue-400 mb-2"><CloudUploadOutline /></n-icon>
              <p class="text-sm text-gray-600 dark:text-gray-400">点击或拖拽文件到此处</p>
              <p class="text-xs text-gray-400 mt-1">支持 PDF、Word、PPT、Excel、TXT、Markdown</p>
              <p class="text-xs text-amber-500/90 dark:text-amber-400/90 mt-2">上传后自动打开编辑页审阅，可直接修改内容；点击「保存入库」才正式进入知识库，取消则保持草稿。</p>
            </div>
          </n-upload-dragger>
        </n-upload>
        <div class="mt-3">
          <div class="text-xs text-gray-500 mb-1.5">标签（可选，用于检索软加权）</div>
          <n-select
            v-model:value="uploadTags" :options="tagSelectOptions"
            multiple filterable tag clearable :disabled="uploading"
            placeholder="选择已有标签，或输入后回车新建" size="small"
          />
        </div>
      </div>
      <template #footer>
        <div class="flex justify-end gap-2">
          <n-button :disabled="uploading" @click="showUpload = false">取消</n-button>
          <n-button type="primary" :loading="uploading" @click="submitUpload">上传</n-button>
        </div>
      </template>
    </AppModal>

    <!-- Image upload modal -->
    <AppModal v-model:show="showImageUpload" title="上传图片" width="min(90vw, 480px)" :loading="imageUploading">
      <div class="relative">
        <div
          v-if="imageUploading"
          class="absolute inset-0 z-10 flex flex-col items-center justify-center gap-3 bg-white/85 dark:bg-gray-800/85 backdrop-blur-sm rounded-lg"
        >
          <n-spin size="large" />
          <span class="text-sm text-gray-600 dark:text-gray-300 font-medium">{{ imageUploadStatus }}</span>
        </div>
        <n-upload
          multiple :custom-request="handleUpload"
          accept=".png,.jpg,.jpeg,.webp,.gif,.bmp"
          :default-upload="false"
          v-model:file-list="imageFileList"
          list-type="image"
          :disabled="imageUploading"
        >
          <n-upload-dragger>
            <div class="py-6 text-center">
              <n-icon :size="36" class="text-orange-400 mb-2"><ImageOutline /></n-icon>
              <p class="text-sm text-gray-600 dark:text-gray-400">点击或拖拽图片到此处</p>
              <p class="text-xs text-gray-400 mt-1">支持 PNG、JPG、WEBP、GIF、BMP · 由多模态模型识别为可编辑文本</p>
            </div>
          </n-upload-dragger>
        </n-upload>
        <p class="text-xs text-gray-400 mt-3 leading-relaxed">
          {{ canUpdateDocument
            ? '上传后由视觉模型转写并自动打开编辑页，可对照原图修改识别文本；点击「保存入库」才正式入库。'
            : '上传后由视觉模型转写并暂存为草稿；点击「保存入库」才正式入库。' }}
        </p>
        <div class="mt-3">
          <div class="text-xs text-gray-500 mb-1.5">标签（可选，用于检索软加权）</div>
          <n-select
            v-model:value="imageUploadTags" :options="tagSelectOptions"
            multiple filterable tag clearable :disabled="imageUploading"
            placeholder="选择已有标签，或输入后回车新建" size="small"
          />
        </div>
      </div>
      <template #footer>
        <div class="flex justify-end gap-2">
          <n-button :disabled="imageUploading" @click="showImageUpload = false">取消</n-button>
          <n-button type="primary" :loading="imageUploading" @click="submitImageUpload">上传识别</n-button>
        </div>
      </template>
    </AppModal>

    <!-- Markdown text editor modal -->
    <n-modal
      v-model:show="showTextEditor"
      to="#app"
      :mask-closable="false"
      :close-on-esc="!submittingText"
    >
      <div
        class="document-editor-modal flex flex-col bg-white dark:bg-gray-800 rounded-[var(--ui-radius-card)] shadow-2xl overflow-hidden"
        style="width: 92vw; max-width: 1240px; height: 86vh"
      >
        <!-- Header -->
        <div class="flex items-center justify-between px-4 sm:px-6 py-3.5 border-b border-gray-200 dark:border-gray-700 shrink-0">
          <div class="flex min-w-0 items-center gap-2">
            <span class="truncate text-base font-semibold text-gray-800 dark:text-gray-100">{{ documentEditorTitle }}</span>
            <n-tag v-if="isViewingDocument" size="small" :bordered="false" round>仅查看</n-tag>
            <n-tag v-if="isDraftEditor" size="small" type="warning" :bordered="false" round>待入库</n-tag>
          </div>
          <n-button text aria-label="关闭文档编辑器" :disabled="submittingText" @click="showTextEditor = false">
            <template #icon><n-icon :size="20"><CloseOutline /></n-icon></template>
          </n-button>
        </div>

        <!-- Processing overlay -->
        <div
          v-if="submittingText"
          class="absolute inset-0 z-10 flex flex-col items-center justify-center gap-4 bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm rounded-[var(--ui-radius-card)]"
        >
          <n-spin size="large" />
          <span class="text-sm text-gray-600 dark:text-gray-300 font-medium">{{ processingStatus }}</span>
        </div>

        <!-- Body -->
        <div class="flex-1 min-h-0 flex flex-col gap-3 px-4 sm:px-6 py-4">
          <n-input
            v-model:value="textTitle"
            placeholder="文档标题"
            class="shrink-0"
            size="large"
            :readonly="isViewingDocument"
          />

          <!-- 数据来源链接：开启后，问答检索的参考来源会把该文档标题渲染成可点击外链 -->
          <div class="flex flex-col gap-2 sm:flex-row sm:items-center sm:gap-3 shrink-0">
            <span class="sm:w-24 shrink-0 text-sm text-gray-600 dark:text-gray-300">数据来源链接</span>
            <n-switch v-model:value="sourceUrlEnabled" size="small" :disabled="isViewingDocument" />
            <n-input
              v-if="sourceUrlEnabled"
              v-model:value="sourceUrl"
              placeholder="https:// 原文链接，问答来源将显示为可点击标题"
              size="small"
              class="w-full sm:flex-1"
              :readonly="isViewingDocument"
            />
          </div>

          <!-- 文档标签：问答时用户勾选这些标签会让本文档命中的片段排序上浮（软加权） -->
          <div class="flex flex-col gap-2 sm:flex-row sm:items-center sm:gap-3 shrink-0">
            <span class="sm:w-24 shrink-0 text-sm text-gray-600 dark:text-gray-300">标签</span>
            <n-select
              v-model:value="editTags" :options="tagSelectOptions"
              multiple filterable tag clearable size="small"
              placeholder="选择已有标签，或输入后回车新建" class="w-full sm:flex-1"
              :disabled="isViewingDocument"
            />
          </div>

          <!-- 原图对照：上传图片转写而来的文档，展示原图便于校对识别结果 -->
          <div v-if="editingImageUrl" class="flex flex-wrap items-center gap-3 shrink-0">
            <span class="sm:w-24 shrink-0 text-sm text-gray-600 dark:text-gray-300">原图对照</span>
            <n-image
              :src="editingImageUrl" width="56" height="56" object-fit="cover"
              class="rounded border border-gray-200 dark:border-gray-700"
            />
            <span class="text-xs text-gray-400">点击图片可放大，对照原图校对识别结果</span>
          </div>

          <div class="flex lg:hidden shrink-0 rounded-[var(--ui-radius-control)] border border-gray-200 dark:border-gray-700 p-1 text-xs">
            <button
              type="button"
              class="flex-1 rounded-lg px-3 py-2 transition-colors"
              :class="mobileEditorPane === 'editor' ? 'bg-blue-500 text-white shadow-sm' : 'text-gray-500 dark:text-gray-400'"
              :aria-pressed="mobileEditorPane === 'editor'"
              @click="mobileEditorPane = 'editor'"
            >{{ isViewingDocument ? '内容' : '编辑' }}</button>
            <button
              type="button"
              class="flex-1 rounded-lg px-3 py-2 transition-colors"
              :class="mobileEditorPane === 'preview' ? 'bg-blue-500 text-white shadow-sm' : 'text-gray-500 dark:text-gray-400'"
              :aria-pressed="mobileEditorPane === 'preview'"
              @click="mobileEditorPane = 'preview'"
            >预览</button>
          </div>

          <div class="grid grid-cols-1 lg:grid-cols-2 gap-4 flex-1 min-h-0">
            <!-- Editor -->
            <div :class="[mobileEditorPane === 'editor' ? 'flex' : 'hidden', 'lg:flex', 'flex-col min-h-0']">
              <div class="flex items-center gap-1.5 text-xs font-medium text-gray-500 mb-1.5 px-1 shrink-0">
                <span class="w-1.5 h-1.5 rounded-full bg-blue-500"></span>{{ isViewingDocument ? 'Markdown 内容' : 'Markdown 编辑' }}
              </div>
              <textarea
                v-model="textContent"
                spellcheck="false"
                :readonly="isViewingDocument"
                class="flex-1 min-h-0 w-full resize-none rounded-lg border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900 text-gray-800 dark:text-gray-200 text-sm leading-relaxed font-mono p-4 outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                placeholder="# 标题&#10;&#10;在此输入 Markdown 内容，支持标题、列表、代码块、表格等..."
              />
            </div>
            <!-- Preview -->
            <div :class="[mobileEditorPane === 'preview' ? 'flex' : 'hidden', 'lg:flex', 'flex-col min-h-0']">
              <div class="flex items-center justify-between mb-1.5 px-1 shrink-0">
                <div class="flex items-center gap-1.5 text-xs font-medium text-gray-500">
                  <span class="w-1.5 h-1.5 rounded-full" :class="previewMode === 'markdown' ? 'bg-green-500' : 'bg-purple-500'"></span>
                  {{ previewMode === 'markdown' ? '实时预览' : `分块预览（${simulatedChunks.length} 块）` }}
                </div>
                <div class="flex items-center rounded border border-gray-200 dark:border-gray-700 overflow-hidden text-xs">
                  <button
                    class="px-2 py-0.5 transition-colors"
                    :class="previewMode === 'markdown' ? 'bg-green-500 text-white' : 'text-gray-500 hover:bg-gray-100 dark:hover:bg-gray-700'"
                    @click="previewMode = 'markdown'"
                  >预览</button>
                  <button
                    class="px-2 py-0.5 transition-colors border-l border-gray-200 dark:border-gray-700"
                    :class="previewMode === 'chunks' ? 'bg-purple-500 text-white' : 'text-gray-500 hover:bg-gray-100 dark:hover:bg-gray-700'"
                    @click="previewMode = 'chunks'"
                  >分块</button>
                </div>
              </div>
              <!-- Markdown preview -->
              <div
                v-if="previewMode === 'markdown'"
                class="markdown-body flex-1 min-h-0 overflow-y-auto rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 text-gray-800 dark:text-gray-200 p-4"
                v-html="renderedMarkdown"
              />
              <!-- Chunks preview -->
              <div
                v-else
                class="flex-1 min-h-0 overflow-y-auto rounded-lg border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900 p-3 space-y-2"
              >
                <div v-if="!simulatedChunks.length" class="text-xs text-gray-400 text-center py-8">暂无内容</div>
                <div
                  v-for="(chunk, i) in simulatedChunks" :key="i"
                  class="bg-white dark:bg-gray-800 rounded-lg border border-purple-200 dark:border-purple-900/50 p-3"
                >
                  <div class="flex items-center gap-2 mb-1.5">
                    <span class="text-xs font-semibold text-purple-600 dark:text-purple-400 bg-purple-50 dark:bg-purple-900/30 px-1.5 py-0.5 rounded font-mono">
                      #{{ i + 1 }}
                    </span>
                    <span class="text-xs text-gray-400">{{ chunk.length }} 字符</span>
                  </div>
                  <p class="text-xs text-gray-700 dark:text-gray-300 leading-relaxed whitespace-pre-wrap font-mono">{{ chunk }}</p>
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- Footer -->
        <div class="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between px-4 sm:px-6 py-3 border-t border-gray-200 dark:border-gray-700 shrink-0">
          <span class="text-xs text-gray-400">
            {{ textContent.length }} 字符 · {{ editorFooterHint }}
          </span>
          <div class="flex justify-end gap-2">
            <n-button :disabled="submittingText" @click="showTextEditor = false">{{ isViewingDocument ? '关闭' : '取消' }}</n-button>
            <n-button
              v-if="!isViewingDocument"
              type="primary"
              :loading="submittingText"
              :disabled="!textTitle.trim() || !textContent.trim()"
              @click="submitText"
            >
              保存入库
            </n-button>
          </div>
        </div>
      </div>
    </n-modal>

    <!-- 标签快捷编辑：仅改标签，不重新解析/嵌入文档 -->
    <AppModal v-model:show="showTagEditor" title="编辑标签" width="min(90vw, 440px)" :loading="savingTags">
      <div class="space-y-2">
        <div class="text-xs text-gray-500">
          {{ tagEditDoc?.filename }}
        </div>
        <n-select
          v-model:value="tagEditValue" :options="tagSelectOptions"
          multiple filterable tag clearable
          placeholder="选择已有标签，或输入后回车新建"
        />
        <p class="text-xs text-gray-400 leading-relaxed">
          标签用于问答检索的软加权：用户勾选某标签时，带该标签的文档命中片段会优先排序，但不会排除其他文档。修改标签不会重新处理文档内容。
        </p>
      </div>
      <template #footer>
        <div class="flex justify-end gap-2">
          <n-button :disabled="savingTags" @click="showTagEditor = false">取消</n-button>
          <n-button type="primary" :loading="savingTags" @click="saveTags">保存</n-button>
        </div>
      </template>
    </AppModal>

    <DangerConfirm
      v-model:show="showDeleteConfirm"
      :title="deleteConfirmTitle"
      :subject="deleteConfirmSubject"
      :description="deleteConfirmDescription"
      :loading="deleting"
      @confirm="confirmDelete"
      @cancel="deleteTarget = null"
    />
    </div>
  </div>
</template>

<script setup>
import { ref, reactive, computed, onMounted, onUnmounted, watch, h } from 'vue'
import { useRoute } from 'vue-router'
import { NButton, NIcon, NSelect, NDataTable, NModal, NUpload, NUploadDragger, NTag, NInput, NSpin, NSwitch, NImage, useMessage } from 'naive-ui'
import { CloudUploadOutline, CloudDoneOutline, TrashOutline, CreateOutline, CloseOutline, PencilOutline, LibraryOutline, ArrowBackOutline, ImageOutline, PricetagsOutline, EyeOutline } from '@vicons/ionicons5'
import { renderDocMarkdown } from '@/utils/markdown'
import { canReadDocumentRow, canUpdateDocumentRow, canDeleteDocumentRow } from '@/utils/documentPermissions'
import { useKnowledgeStore } from '@/stores/knowledge'
import { useAuthStore } from '@/stores/auth'
import { getAllDocuments, uploadDocument, uploadImageDocument, ingestDocument, deleteDocument, toggleDocument, createTextDocument, getDocument, getDocumentImage, updateTextDocument, updateDocumentTags } from '@/api/document'
import { getDocumentTags } from '@/api/knowledge'
import PageHeader from '@/components/ui/PageHeader.vue'
import SurfaceCard from '@/components/ui/SurfaceCard.vue'
import RowActions from '@/components/ui/RowActions.vue'
import DangerConfirm from '@/components/ui/DangerConfirm.vue'
import AppModal from '@/components/ui/AppModal.vue'

const route = useRoute()
const kbStore = useKnowledgeStore()
const authStore = useAuthStore()
const msg = useMessage()
const canReadDocument = computed(() => authStore.hasPerm('doc:read'))
const canCreateDocument = computed(() => authStore.hasPerm('doc:create'))
const canUpdateDocument = computed(() => authStore.hasPerm('doc:update'))
const canDeleteDocument = computed(() => authStore.hasPerm('doc:delete'))
const selectedKbId = ref(null)
const docs = ref([])
const loading = ref(false)
const checkedRowKeys = ref([])
const rowKey = (row) => row.id
const showDeleteConfirm = ref(false)
const deleteTarget = ref(null)
const deleting = ref(false)
const ingestingId = ref(null)
const batchIngesting = ref(false)
const deleteConfirmTitle = computed(() => deleteTarget.value?.kind === 'batch' ? '删除选中文档？' : '删除文档？')
const deleteConfirmSubject = computed(() => {
  if (!deleteTarget.value) return ''
  if (deleteTarget.value.kind === 'batch') return `已选择 ${deleteTarget.value.ids.length} 个文档`
  return deleteTarget.value.row.filename
})
const deleteConfirmDescription = computed(() => deleteTarget.value?.kind === 'batch'
  ? '删除后，所选文档及其已生成的检索内容都无法恢复。'
  : '删除后，该文档及其已生成的检索内容都无法恢复。'
)
// 勾选行里可以由当前用户“保存入库”的草稿数
const checkedDraftCount = computed(() => checkedRowKeys.value.filter(id => {
  const row = docs.value.find(d => d.id === id)
  return row?.status === 'draft' && canUpdateDocumentRow(row)
}).length)
const pagination = reactive({
  page: 1,
  pageSize: 10,                      // 默认每页 10 条
  showSizePicker: true,              // 显示每页条数选择器
  pageSizes: [10, 20, 30, 50],       // 可自定义选择
  prefix: ({ itemCount }) => `共 ${itemCount} 条`,
  onUpdatePage: (p) => { pagination.page = p },
  onUpdatePageSize: (ps) => { pagination.pageSize = ps; pagination.page = 1 },
})
// 删除导致总数变化时，若当前页超出范围则回退到最后一页
watch(() => docs.value.length, () => {
  const max = Math.max(1, Math.ceil(docs.value.length / pagination.pageSize))
  if (pagination.page > max) pagination.page = max
})

// 当前知识库已有的标签，用作各处标签输入框的候选项（也允许现场新建）
const kbTags = ref([])
const tagSelectOptions = computed(() => kbTags.value.map(t => ({ label: t, value: t })))

async function loadKbTags() {
  if (!selectedKbId.value) { kbTags.value = []; return }
  try { kbTags.value = await getDocumentTags([selectedKbId.value]) }
  catch { kbTags.value = [] }
}

// upload
const showUpload = ref(false)
const uploading = ref(false)
const uploadStatus = ref('')
const fileList = ref([])
const uploadTags = ref([])

// image upload
const showImageUpload = ref(false)
const imageUploading = ref(false)
const imageUploadStatus = ref('')
const imageFileList = ref([])
const imageUploadTags = ref([])

// 标签快捷编辑（不触发重新解析/嵌入）
const showTagEditor = ref(false)
const tagEditDoc = ref(null)
const tagEditValue = ref([])
const savingTags = ref(false)

// text editor
const showTextEditor = ref(false)
const textTitle = ref('')
const textContent = ref('')
const submittingText = ref(false)
const processingStatus = ref('')
const editingDocId = ref(null)
const editorMode = ref('create')
const editingImageUrl = ref(null)
const sourceUrlEnabled = ref(false)
const sourceUrl = ref('')
const editTags = ref([])
const editingDocument = ref(null)

const renderedMarkdown = computed(() => renderDocMarkdown(textContent.value || ''))
const isViewingDocument = computed(() => editorMode.value === 'view')
const isDraftEditor = computed(() => editorMode.value === 'draft')
const documentEditorTitle = computed(() => ({
  create: '手动输入文档',
  edit: '编辑文档',
  view: '查看文档',
  draft: '审阅文档 · 待入库',
})[editorMode.value] || '文档内容')
const editorFooterHint = computed(() => {
  if (isViewingDocument.value) return '当前仅查看'
  if (isDraftEditor.value) return '草稿内容 · 编辑后保存入库'
  return '按标题自动分块入库'
})
const IMAGE_EXTS = new Set(['png', 'jpg', 'jpeg', 'webp', 'gif', 'bmp'])

const previewMode = ref('markdown')
const mobileEditorPane = ref('editor')

function clearEditingImageUrl() {
  if (editingImageUrl.value?.startsWith('blob:')) {
    URL.revokeObjectURL(editingImageUrl.value)
  }
  editingImageUrl.value = null
}

watch(showTextEditor, (shown) => {
  if (!shown) {
    clearEditingImageUrl()
    editingDocument.value = null
  }
})
onUnmounted(clearEditingImageUrl)

function _splitText(text) {
  const CHUNK_SIZE = 500
  const CHUNK_OVERLAP = 50
  text = text.replace(/\n{3,}/g, '\n\n').trim()
  if (!text) return []
  const words = text.split(/\s+/).filter(Boolean)
  const useCharSplit = !words.length || (text.length / words.length) > 4
  if (useCharSplit) {
    const chunks = []
    const step = CHUNK_SIZE - CHUNK_OVERLAP
    for (let i = 0; i < text.length; i += step) {
      const start = i > 0 ? Math.max(0, i - CHUNK_OVERLAP) : 0
      const chunk = text.slice(start, i + CHUNK_SIZE)
      if (chunk.trim()) chunks.push(chunk)
    }
    return chunks.filter(c => c.trim().length > 5)
  }
  const chunks = []
  let current = []
  let count = 0
  for (const word of words) {
    current.push(word)
    count++
    if (count >= CHUNK_SIZE) {
      chunks.push(current.join(' '))
      current = current.slice(-CHUNK_OVERLAP)
      count = current.length
    }
  }
  if (current.length) chunks.push(current.join(' '))
  return chunks.filter(c => c.trim().length > 20)
}

const simulatedChunks = computed(() => {
  const content = textContent.value
  if (!content) return []
  const sectionRe = /^(#{1,6}\s.+)$/gm
  const parts = content.split(sectionRe)
  const sections = []
  let cur = ''
  for (const part of parts) {
    if (/^#{1,6}\s/.test(part)) {
      if (cur.trim()) sections.push(cur.trim())
      cur = part + '\n'
    } else {
      cur += part
    }
  }
  if (cur.trim()) sections.push(cur.trim())
  if (!sections.length) sections.push(content)
  return sections.flatMap(s => _splitText(s))
})

const kbOptions = computed(() => kbStore.list.map(kb => ({ label: kb.name, value: kb.id })))

const statusTag = (s) => {
  const map = { ready: ['success', '就绪'], processing: ['warning', '处理中'], draft: ['default', '草稿'], failed: ['error', '失败'] }
  const [type, text] = map[s] || ['default', s]
  return h(NTag, { type, size: 'small' }, () => text)
}

const fmtTime = value => value ? new Intl.DateTimeFormat('zh-CN', {
  year: 'numeric', month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', second: '2-digit', hourCycle: 'h23',
}).format(new Date(value)).replaceAll('/', '-') : '—'

const columns = computed(() => [
  ...(canDeleteDocument.value ? [{
    type: 'selection',
    align: 'center',
    titleAlign: 'center',
    disabled: row => !canDeleteDocumentRow(row),
  }] : []),
  { title: '文件名', key: 'filename', minWidth: 190, align: 'left', titleAlign: 'left', ellipsis: { tooltip: true } },
  {
    title: '标签', key: 'tags', minWidth: 150, align: 'left', titleAlign: 'left',
    render: row => (row.tags && row.tags.length)
      ? h('div', { style: 'display:flex;flex-wrap:wrap;gap:4px' },
          row.tags.map(t => h(NTag, { size: 'small', type: 'info', bordered: false }, () => t)))
      : h('span', { class: 'text-xs text-gray-400' }, '—')
  },
  { title: '类型', key: 'file_type', width: 82, align: 'center', titleAlign: 'center' },
  { title: '分块数', key: 'chunk_count', width: 82, align: 'center', titleAlign: 'center' },
  { title: '状态', key: 'status', width: 96, align: 'center', titleAlign: 'center', render: row => row.status === 'draft' ? statusTag('draft') : (row.is_active ? statusTag(row.status) : h(NTag, { type: 'default', size: 'small' }, () => '停用')) },
  ...(canUpdateDocument.value ? [{
    title: '启用', key: 'is_active', width: 80, align: 'center', titleAlign: 'center',
    render: row => h(NSwitch, {
      value: row.is_active,
      size: 'small',
      disabled: row.status === 'draft' || !canUpdateDocumentRow(row),
      title: canUpdateDocumentRow(row) ? '切换文档启用状态' : '仅文档创建者或超级管理员可以修改',
      onUpdateValue: () => handleToggle(row),
    })
  }] : []),
  { title: '上传时间', key: 'created_at', width: 168, align: 'center', titleAlign: 'center', render: r => fmtTime(r.created_at) },
  // 未实际修改过时，修改时间默认取创建时间
  { title: '最近修改', key: 'updated_at', width: 168, align: 'center', titleAlign: 'center', render: r => fmtTime(r.updated_at || r.created_at) },
  { title: '创建人', key: 'created_by_name', width: 100, align: 'left', titleAlign: 'left', render: r => r.created_by_name || '—' },
  // 未实际修改过时，修改人默认取创建人
  { title: '修改人', key: 'updated_by_name', width: 100, align: 'left', titleAlign: 'left', render: r => r.updated_by_name || r.created_by_name || '—' },
  ...(canReadDocument.value || canUpdateDocument.value || canDeleteDocument.value ? [{
    title: '操作', key: 'actions', width: 132, fixed: 'right', align: 'center', titleAlign: 'center',
    render: row => h(RowActions, { label: `文档 ${row.filename} 操作` }, {
      default: () => [
        row.status === 'draft' && canReadDocumentRow(row)
          ? h(NButton, { text: true, size: 'small', 'aria-label': '审阅草稿', title: '审阅草稿', onClick: () => openDraftEditor(row) },
              { icon: () => h(NIcon, null, () => h(EyeOutline)) })
          : null,
        row.status === 'draft'
          ? (canUpdateDocumentRow(row)
              ? h(NButton, { text: true, type: 'primary', size: 'small', 'aria-label': '保存入库', title: '保存入库', loading: ingestingId.value === row.id, onClick: () => submitIngest(row) },
                  { icon: () => h(NIcon, null, () => h(CloudDoneOutline)) })
              : null)
          : null,
        row.status !== 'draft' && canReadDocumentRow(row)
          ? h(NButton, { text: true, size: 'small', 'aria-label': '查看文档', title: '查看文档', onClick: () => openViewEditor(row) },
              { icon: () => h(NIcon, null, () => h(EyeOutline)) })
          : null,
        row.status !== 'draft' && canUpdateDocumentRow(row)
          ? h(NButton, { text: true, type: 'primary', size: 'small', 'aria-label': '编辑标签', title: '编辑标签', onClick: () => openTagEditor(row) },
              { icon: () => h(NIcon, null, () => h(PricetagsOutline)) })
          : null,
        row.status !== 'draft' && canUpdateDocumentRow(row)
          ? h(NButton, { text: true, type: 'primary', size: 'small', 'aria-label': '编辑内容', title: '编辑内容', onClick: () => openEditEditor(row) },
              { icon: () => h(NIcon, null, () => h(PencilOutline)) })
          : null,
        canDeleteDocumentRow(row)
          ? h(NButton, { text: true, type: 'error', size: 'small', 'aria-label': '删除文档', title: '删除文档', onClick: () => openDelete(row) },
              { icon: () => h(NIcon, null, () => h(TrashOutline)) })
          : null,
      ].filter(Boolean),
    })
  }] : [])
])

onMounted(async () => {
  await kbStore.fetchList()
  if (route.query.kb) selectedKbId.value = route.query.kb
})

watch(selectedKbId, async (id) => {
  checkedRowKeys.value = []
  if (!id) { docs.value = []; pagination.page = 1; kbTags.value = []; return }
  if (!canReadDocument.value) {
    docs.value = []
    kbTags.value = []
    msg.warning('当前角色没有查看文档的权限')
    return
  }
  loading.value = true
  try { docs.value = await getAllDocuments(id); pagination.page = 1; await loadKbTags() }
  finally { loading.value = false }
})

async function handleToggle(row) {
  if (!canUpdateDocumentRow(row)) {
    msg.warning('只有文档创建者或超级管理员可以修改该文档')
    return
  }
  const updated = await toggleDocument(selectedKbId.value, row.id)
  const idx = docs.value.findIndex(d => d.id === row.id)
  if (idx !== -1) docs.value[idx] = { ...docs.value[idx], ...updated }
  msg.success(updated.is_active ? '文档已启用' : '文档已停用')
}

function openDelete(row) {
  if (!canDeleteDocumentRow(row)) {
    msg.warning('只有文档创建者或超级管理员可以删除该文档')
    return
  }
  deleteTarget.value = { kind: 'single', row }
  showDeleteConfirm.value = true
}

function openBatchDelete() {
  if (!canDeleteDocument.value) {
    msg.warning('当前角色没有删除文档的权限')
    return
  }
  const ids = checkedRowKeys.value.filter(id => {
    const document = docs.value.find(row => row.id === id)
    return canDeleteDocumentRow(document)
  })
  if (!ids.length) return
  deleteTarget.value = { kind: 'batch', ids }
  showDeleteConfirm.value = true
}

async function confirmDelete() {
  const target = deleteTarget.value
  if (!target || !selectedKbId.value) return
  if (!canDeleteDocument.value) {
    msg.warning('当前角色没有删除文档的权限')
    return
  }
  if (target.kind === 'single' && !canDeleteDocumentRow(target.row)) {
    msg.warning('只有文档创建者或超级管理员可以删除该文档')
    return
  }
  deleting.value = true
  try {
    if (target.kind === 'single') {
      await deleteDocument(selectedKbId.value, target.row.id)
      docs.value = docs.value.filter(d => d.id !== target.row.id)
      checkedRowKeys.value = checkedRowKeys.value.filter(id => id !== target.row.id)
      msg.success('文档已删除')
    } else {
      const ids = target.ids
      const results = await Promise.allSettled(ids.map(id => deleteDocument(selectedKbId.value, id)))
      const okIds = ids.filter((_, i) => results[i].status === 'fulfilled')
      const okSet = new Set(okIds)
      docs.value = docs.value.filter(d => !okSet.has(d.id))
      checkedRowKeys.value = checkedRowKeys.value.filter(id => !okSet.has(id))
      const failed = ids.length - okIds.length
      if (failed) msg.warning(`${okIds.length} 个已删除，${failed} 个删除失败`)
      else msg.success(`已删除 ${okIds.length} 个文档`)
    }
    await loadKbTags()
    showDeleteConfirm.value = false
    deleteTarget.value = null
  } catch (error) {
    msg.error(error?.response?.data?.detail || '删除失败，请重试')
  } finally {
    deleting.value = false
  }
}

async function submitUpload() {
  if (!fileList.value.length || !selectedKbId.value) return
  if (!canCreateDocument.value) {
    msg.warning('当前角色没有新增文档的权限')
    return
  }
  uploading.value = true
  uploadStatus.value = '正在上传文件...'
  try {
    const uploaded = []
    for (let i = 0; i < fileList.value.length; i++) {
      uploadStatus.value = `正在上传 ${i + 1} / ${fileList.value.length}...`
      const doc = await uploadDocument(selectedKbId.value, fileList.value[i].file, null, uploadTags.value)
      uploaded.push(doc)
    }
    // 上传只暂存为草稿并准备审阅内容；分块/向量化由「保存入库」显式触发。
    uploadStatus.value = '正在准备内容...'
    const results = await Promise.allSettled(uploaded.map(doc => pollDocumentPrepared(doc.id)))
    const failed = results.filter(r => r.status === 'rejected').length
    const prepared = uploaded.filter((_, i) => results[i].status === 'fulfilled')
    showUpload.value = false
    fileList.value = []
    uploadTags.value = []
    docs.value = await getAllDocuments(selectedKbId.value)
    await loadKbTags()
    if (failed > 0) {
      msg.warning(`${uploaded.length - failed} 个已暂存为草稿，${failed} 个内容准备失败（可在列表中删除）`)
    } else if (uploaded.length === 1 && prepared[0] && canUpdateDocumentRow(prepared[0])) {
      // 单文件：自动打开审阅编辑器；确认「保存入库」才正式入库，取消则保持草稿
      await openDraftEditor(prepared[0])
    } else {
      msg.success(`${uploaded.length} 个文档已暂存为草稿，可打开审阅后点击「保存入库」`)
    }
  } catch {
    msg.error('上传失败，请重试')
  } finally {
    uploading.value = false
    uploadStatus.value = ''
  }
}

function handleUpload({ file, onFinish }) { onFinish() }

function openTextEditor() {
  if (!canCreateDocument.value) {
    msg.warning('当前角色没有新增文档的权限')
    return
  }
  editingDocId.value = null
  editingDocument.value = null
  editorMode.value = 'create'
  mobileEditorPane.value = 'editor'
  clearEditingImageUrl()
  textTitle.value = ''
  textContent.value = ''
  sourceUrlEnabled.value = false
  sourceUrl.value = ''
  editTags.value = []
  showTextEditor.value = true
}

function openViewEditor(row) {
  return openDocumentEditor(row, 'view')
}

function openEditEditor(row) {
  return openDocumentEditor(row, 'edit')
}

async function openDocumentEditor(row, mode) {
  if (row.status === 'draft' && mode !== 'draft') {
    msg.warning('该文档尚未入库，请先点击「保存入库」')
    return
  }
  if (!canReadDocument.value || !canReadDocumentRow(row)) {
    msg.warning('当前角色没有查看文档的权限')
    return
  }
  if (mode === 'edit' && !canUpdateDocumentRow(row)) {
    msg.warning('只有文档创建者或超级管理员可以修改该文档')
    return
  }
  try {
    const doc = await getDocument(selectedKbId.value, row.id)
    if (mode === 'edit' && !canUpdateDocumentRow(doc)) {
      msg.warning('文档权限已变化，请刷新后重试')
      return
    }
    editingDocId.value = row.id
    editingDocument.value = doc
    editorMode.value = mode === 'edit' ? 'edit' : mode === 'draft' ? 'draft' : 'view'
    mobileEditorPane.value = 'editor'
    clearEditingImageUrl()
    if (doc.image_url) {
      try {
        const imageBlob = await getDocumentImage(doc.image_url)
        editingImageUrl.value = URL.createObjectURL(imageBlob)
      } catch (error) {
        msg.warning(error?.response?.data?.detail || (mode === 'view'
          ? '原图加载失败，仍可继续查看识别文本'
          : '原图加载失败，仍可继续编辑识别文本'))
      }
    }
    textTitle.value = doc.filename
    textContent.value = doc.raw_content || ''
    sourceUrl.value = doc.source_url || ''
    sourceUrlEnabled.value = !!doc.source_url
    editTags.value = [...(doc.tags || [])]
    showTextEditor.value = true
    if (mode === 'draft' && !(doc.raw_content || '').trim()) {
      msg.warning('内容仍在准备中或解析失败，可稍后重试或删除该草稿')
    }
  } catch {
    msg.error('加载文档内容失败')
  }
}

function openDraftEditor(row) {
  return openDocumentEditor(row, 'draft')
}

function openTagEditor(row) {
  if (!canUpdateDocumentRow(row)) {
    msg.warning('只有文档创建者或超级管理员可以修改该文档')
    return
  }
  tagEditDoc.value = row
  tagEditValue.value = [...(row.tags || [])]
  showTagEditor.value = true
}

async function saveTags() {
  if (!tagEditDoc.value) return
  if (!canUpdateDocumentRow(tagEditDoc.value)) {
    msg.warning('只有文档创建者或超级管理员可以修改该文档')
    return
  }
  savingTags.value = true
  try {
    const updated = await updateDocumentTags(selectedKbId.value, tagEditDoc.value.id, tagEditValue.value)
    const idx = docs.value.findIndex(d => d.id === tagEditDoc.value.id)
    if (idx !== -1) docs.value[idx] = { ...docs.value[idx], ...updated }
    msg.success('标签已更新')
    showTagEditor.value = false
    await loadKbTags()
  } catch {
    msg.error('标签更新失败，请重试')
  } finally {
    savingTags.value = false
  }
}

async function submitImageUpload() {
  if (!imageFileList.value.length || !selectedKbId.value) return
  if (!canCreateDocument.value) {
    msg.warning('当前角色没有新增文档的权限')
    return
  }
  imageUploading.value = true
  imageUploadStatus.value = '正在上传图片...'
  try {
    const uploaded = []
    for (let i = 0; i < imageFileList.value.length; i++) {
      imageUploadStatus.value = `正在上传 ${i + 1} / ${imageFileList.value.length}...`
      const doc = await uploadImageDocument(selectedKbId.value, imageFileList.value[i].file, imageUploadTags.value)
      uploaded.push(doc)
    }
    // 上传只暂存为草稿并视觉转写（prepare）；分块/向量化由「保存入库」显式触发。
    imageUploadStatus.value = '视觉模型正在识别图片内容...'
    const results = await Promise.allSettled(uploaded.map(doc => pollDocumentPrepared(doc.id)))
    const failed = results.filter(r => r.status === 'rejected').length
    const prepared = uploaded.filter((_, i) => results[i].status === 'fulfilled')
    showImageUpload.value = false
    imageFileList.value = []
    imageUploadTags.value = []
    docs.value = await getAllDocuments(selectedKbId.value)
    await loadKbTags()
    if (failed > 0) {
      msg.warning(`${uploaded.length - failed} 张已暂存为草稿，${failed} 张识别失败（可在列表中删除，或检查「设置 → 多模态模型」配置）`)
    } else if (uploaded.length === 1 && prepared[0] && canUpdateDocumentRow(prepared[0])) {
      // 单张：自动打开审阅编辑器，对照原图校对识别结果；确认入库或取消保持草稿
      await openDraftEditor(prepared[0])
    } else {
      msg.success(`${uploaded.length} 张图片已暂存为草稿，可打开审阅后点击「保存入库」`)
    }
  } catch {
    msg.error('图片上传失败，请重试')
  } finally {
    imageUploading.value = false
    imageUploadStatus.value = ''
  }
}

function pollDocumentStatus(docId) {
  return new Promise((resolve, reject) => {
    const timer = setInterval(async () => {
      try {
        const doc = await getDocument(selectedKbId.value, docId)
        if (doc.status === 'ready') { clearInterval(timer); resolve(doc) }
        else if (doc.status === 'failed') { clearInterval(timer); reject(new Error('处理失败')) }
      } catch (e) { clearInterval(timer); reject(e) }
    }, 2000)
    setTimeout(() => { clearInterval(timer); reject(new Error('处理超时')) }, 120000)
  })
}

// 草稿的内容准备（prepare 任务）完成后 raw_content 才有值，供编辑页审阅
function pollDocumentPrepared(docId) {
  return new Promise((resolve, reject) => {
    const timer = setInterval(async () => {
      try {
        const doc = await getDocument(selectedKbId.value, docId)
        if (doc.status === 'failed') { clearInterval(timer); reject(new Error('内容准备失败')) }
        else if (doc.status !== 'draft' || (doc.raw_content || '').trim()) { clearInterval(timer); resolve(doc) }
      } catch (e) { clearInterval(timer); reject(e) }
    }, 2000)
    setTimeout(() => { clearInterval(timer); reject(new Error('处理超时')) }, 120000)
  })
}

async function submitIngest(row) {
  if (!selectedKbId.value || !canUpdateDocumentRow(row)) {
    msg.warning('只有文档创建者或超级管理员可以保存入库')
    return
  }
  if (row.status !== 'draft') return
  if (IMAGE_EXTS.has((row.file_type || '').toLowerCase()) && !(row.raw_content || '').trim()) {
    msg.warning('图片内容仍在准备中，请稍候再试')
    return
  }
  ingestingId.value = row.id
  try {
    const doc = await ingestDocument(selectedKbId.value, row.id)
    const ready = await pollDocumentStatus(doc.id)
    const idx = docs.value.findIndex(d => d.id === row.id)
    if (idx !== -1) docs.value[idx] = { ...docs.value[idx], ...ready }
    msg.success('文档已保存入库')
    // 单文档入库后自动打开审阅编辑器，保留“解析后校对”体验
    if (canUpdateDocumentRow(ready)) await openEditEditor(ready)
  } catch (e) {
    msg.error(e.message === '处理失败' ? '文档入库失败，请检查文件内容后重试' : '保存入库失败，请重试')
  } finally {
    ingestingId.value = null
  }
}

async function submitBatchIngest() {
  if (!selectedKbId.value) return
  const targets = checkedRowKeys.value
    .map(id => docs.value.find(d => d.id === id))
    .filter(row => row?.status === 'draft' && canUpdateDocumentRow(row))
  const unprepared = targets.filter(row =>
    IMAGE_EXTS.has((row.file_type || '').toLowerCase()) && !(row.raw_content || '').trim()
  )
  const ready = targets.filter(row => !unprepared.includes(row))
  if (unprepared.length) msg.warning(`${unprepared.length} 张图片内容仍在准备中，已跳过`)
  if (!ready.length) return
  batchIngesting.value = true
  try {
    const results = await Promise.allSettled(ready.map(async row => {
      const doc = await ingestDocument(selectedKbId.value, row.id)
      return pollDocumentStatus(doc.id)
    }))
    const ok = results.filter(r => r.status === 'fulfilled').length
    const failed = ready.length - ok
    docs.value = await getAllDocuments(selectedKbId.value)
    checkedRowKeys.value = []
    if (failed) msg.warning(`${ok} 个已入库，${failed} 个入库失败`)
    else msg.success(`${ok} 个文档已保存入库`)
  } catch {
    msg.error('保存入库失败，请重试')
  } finally {
    batchIngesting.value = false
  }
}

async function submitText() {
  if (!textTitle.value.trim() || !textContent.value.trim()) return
  // 草稿编辑页：点击「保存入库」即正式入库（解析/分块/向量化），取消则保持草稿
  if (editorMode.value === 'draft') {
    if (!editingDocId.value || !canUpdateDocumentRow(editingDocument.value)) {
      msg.warning('只有文档创建者或超级管理员可以保存入库')
      return
    }
    submittingText.value = true
    processingStatus.value = '正在保存入库...'
    try {
      const url = (sourceUrlEnabled.value && sourceUrl.value.trim()) ? sourceUrl.value.trim() : null
      const doc = await ingestDocument(selectedKbId.value, editingDocId.value, {
        title: textTitle.value.trim(),
        content: textContent.value,
        source_url: url,
        tags: editTags.value,
      })
      await pollDocumentStatus(doc.id)
      msg.success('文档已保存入库')
      showTextEditor.value = false
      docs.value = await getAllDocuments(selectedKbId.value)
      await loadKbTags()
    } catch (e) {
      msg.error(e.message === '处理失败' ? '文档入库失败，请检查文件内容后重试' : '保存入库失败，请重试')
    } finally {
      submittingText.value = false
      processingStatus.value = ''
    }
    return
  }
  const isUpdate = editorMode.value === 'edit' && !!editingDocId.value
  if (isViewingDocument.value) {
    msg.warning('当前为只读查看模式')
    return
  }
  if (isUpdate ? !canUpdateDocumentRow(editingDocument.value) : !canCreateDocument.value) {
    msg.warning(isUpdate ? '只有文档创建者或超级管理员可以修改该文档' : '当前角色没有新增文档的权限')
    return
  }
  submittingText.value = true
  processingStatus.value = '正在保存...'
  try {
    const url = (sourceUrlEnabled.value && sourceUrl.value.trim()) ? sourceUrl.value.trim() : null
    let doc
    if (isUpdate) {
      doc = await updateTextDocument(selectedKbId.value, editingDocId.value, textTitle.value.trim(), textContent.value, url, editTags.value)
    } else {
      doc = await createTextDocument(selectedKbId.value, textTitle.value.trim(), textContent.value, url, editTags.value)
    }
    processingStatus.value = '正在分块处理...'
    await pollDocumentStatus(doc.id)
    msg.success(isUpdate ? '文档已更新' : '文档已保存')
    showTextEditor.value = false
    docs.value = await getAllDocuments(selectedKbId.value)
    await loadKbTags()
  } catch (e) {
    msg.error(e.message === '处理失败' ? '文档处理失败，请检查内容后重试' : '保存失败，请重试')
  } finally {
    submittingText.value = false
    processingStatus.value = ''
  }
}
</script>
