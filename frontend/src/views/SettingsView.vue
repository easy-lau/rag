<template>
  <div class="p-4 sm:p-6 h-full overflow-y-auto">
    <n-spin :show="settingsStore.loading">
      <div class="max-w-6xl mx-auto space-y-5">
        <PageHeader
          title="系统设置"
          description="管理检索策略与站点品牌信息。模型服务配置请前往“模型管理”。"
        >
          <template #meta>
            <n-tag :type="canWrite ? 'success' : 'warning'" :bordered="false" round>
              {{ canWrite ? '可编辑' : '仅查看' }}
            </n-tag>
          </template>
        </PageHeader>
        <div v-if="!canWrite" class="mb-5 flex items-center gap-2 rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-800 dark:border-amber-900/70 dark:bg-amber-950/30 dark:text-amber-200">
          <n-icon :size="17"><LockClosedOutline /></n-icon>
          当前账号仅可查看系统设置，不能修改或上传站点资源。
        </div>
        <!-- 检索参数 -->
        <SurfaceCard padding="lg" class="mt-6">
          <h3 class="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-4 flex items-center gap-2">
            <span class="w-2 h-2 rounded-full bg-green-500 inline-block"></span>
            检索参数
          </h3>
          <n-form :model="form" :disabled="!canWrite" label-placement="top">
            <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 items-end">
              <n-form-item label="Top K（召回数量）">
                <n-input-number v-model:value="form.top_k" :min="1" :max="20" class="w-full" />
              </n-form-item>
              <n-form-item>
                <template #label>
                  <span class="inline-flex items-center gap-1">
                    默认开启重排
                    <n-tooltip trigger="hover" placement="top">
                      <template #trigger>
                        <n-icon :size="15" class="text-gray-400 cursor-help"><HelpCircleOutline /></n-icon>
                      </template>
                      <div class="max-w-xs text-xs leading-relaxed">
                        重排（Rerank）：先快速召回一批候选片段，再用大模型逐条评估它们与问题的相关度并重新排序，把最相关的排前、剔除不相关的。<br>
                        · 开启：回答更精准、来源更干净，但每次问答多一次模型调用，略慢、成本略增。<br>
                        · 关闭：直接用初步检索结果，更快更省，但可能掺入不相关内容。
                      </div>
                    </n-tooltip>
                  </span>
                </template>
                <n-switch v-model:value="form.rerank_enabled" />
              </n-form-item>
              <n-form-item>
                <template #label>
                  <span class="inline-flex items-center gap-1">
                    显示参考来源
                    <n-tooltip trigger="hover" placement="top">
                      <template #trigger>
                        <n-icon :size="15" class="text-gray-400 cursor-help"><HelpCircleOutline /></n-icon>
                      </template>
                      <div class="max-w-xs text-xs leading-relaxed">
                        开启后，回答下方会区分展示「回答依据」「相近资料」和文档外部链接；关闭后不显示来源（历史记录仍会保留，重新开启即可再次查看）。
                      </div>
                    </n-tooltip>
                  </span>
                </template>
                <n-switch v-model:value="form.show_sources" />
              </n-form-item>
              <n-form-item>
                <template #label>
                  <span class="inline-flex items-center gap-1">
                    知识库未命中策略
                    <n-tooltip trigger="hover" placement="top">
                      <template #trigger>
                        <n-icon :size="15" class="text-gray-400 cursor-help"><HelpCircleOutline /></n-icon>
                      </template>
                      <div class="max-w-xs text-xs leading-relaxed">
                        通用大模型回答不会被标记为知识库命中，也不会展示或保存为回答依据。<br>
                        · 严格模式：未命中时只提示没有知识库依据。<br>
                        · 仅完全未命中：知识库没有可用资料时自动使用通用大模型。<br>
                        · 未命中或证据不足：相关资料不足以闭合答案时也允许通用大模型回答。
                      </div>
                    </n-tooltip>
                  </span>
                </template>
                <n-select
                  v-model:value="form.rag_general_fallback_mode"
                  :options="generalFallbackOptions"
                  class="w-full"
                />
              </n-form-item>
              <n-form-item label="兜底低延迟模型">
                <n-input
                  v-model:value="form.rag_general_fallback_model"
                  placeholder="留空则使用主对话模型"
                  clearable
                />
              </n-form-item>
            </div>
          </n-form>
        </SurfaceCard>

        <!-- 站点设置 -->
        <SurfaceCard padding="lg" class="mt-6">
          <h3 class="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-1 flex items-center gap-2">
            <span class="w-2 h-2 rounded-full bg-pink-500 inline-block"></span>
            站点设置
          </h3>
          <p class="text-xs text-gray-400 mb-4">配置左上角的标题、图标、描述，浏览器标签标题，以及页面底部版权（所有人可见）</p>
          <n-form :model="form" :disabled="!canWrite" label-placement="top">
            <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <n-form-item label="网站标题">
                <n-input v-model:value="form.site_title" placeholder="RAG 检索系统" />
              </n-form-item>
              <n-form-item label="网站描述">
                <n-input v-model:value="form.site_description" placeholder="知识增强·精准问答" />
              </n-form-item>
              <n-form-item label="浏览器标题">
                <n-input v-model:value="form.browser_title" placeholder="留空则使用网站标题" />
              </n-form-item>
              <n-form-item label="底部版权">
                <n-input v-model:value="form.site_copyright" placeholder="如：© 2026 公司名称 版权所有（留空则不显示页脚）" />
              </n-form-item>
              <n-form-item label="网站图标">
                <div class="flex items-center gap-3">
                  <img
                    v-if="form.site_logo"
                    :src="form.site_logo"
                    class="w-10 h-10 rounded-lg object-cover border border-gray-200 dark:border-gray-700"
                    alt="logo"
                  />
                  <n-upload
                    :show-file-list="false"
                    accept="image/png,image/jpeg,image/webp,image/gif,image/svg+xml,image/x-icon,.ico"
                    :custom-request="handleLogoUpload"
                    :disabled="!canWrite"
                  >
                    <n-button size="small" :disabled="!canWrite">{{ form.site_logo ? '更换图标' : '上传图标' }}</n-button>
                  </n-upload>
                  <n-button v-if="form.site_logo" text size="small" type="error" :disabled="!canWrite" @click="form.site_logo = ''">清除</n-button>
                </div>
              </n-form-item>
            </div>
          </n-form>
        </SurfaceCard>

        <div v-if="canWrite" class="flex justify-end mt-6">
          <n-button type="primary" size="large" class="px-10" :loading="saving" @click="handleSave">保存设置</n-button>
        </div>
      </div>
    </n-spin>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { NForm, NFormItem, NInput, NInputNumber, NSelect, NSwitch, NButton, NSpin, NTooltip, NIcon, NTag, NUpload, useMessage } from 'naive-ui'
import { HelpCircleOutline, LockClosedOutline } from '@vicons/ionicons5'
import { useSettingsStore } from '@/stores/settings'
import { useAuthStore } from '@/stores/auth'
import { useSiteStore } from '@/stores/site'
import { uploadLogo } from '@/api/settings'
import PageHeader from '@/components/ui/PageHeader.vue'
import SurfaceCard from '@/components/ui/SurfaceCard.vue'

const settingsStore = useSettingsStore()
const authStore = useAuthStore()
const siteStore = useSiteStore()
const msg = useMessage()
const saving = ref(false)
const form = ref({ ...settingsStore.data })
const canWrite = computed(() => authStore.hasPerm('settings:write'))
const generalFallbackOptions = [
  { label: '严格知识库模式', value: 'off' },
  { label: '仅完全未命中时兜底', value: 'no_hit' },
  { label: '未命中或证据不足时兜底', value: 'no_hit_or_insufficient' },
]

onMounted(async () => {
  await settingsStore.fetch()
  form.value = { ...settingsStore.data }
})

async function handleLogoUpload({ file, onFinish, onError }) {
  if (!canWrite.value) {
    onError()
    return
  }
  try {
    const { url } = await uploadLogo(file.file)
    form.value.site_logo = url
    msg.success('图标已上传')
    onFinish()
  } catch {
    msg.error('图标上传失败')
    onError()
  }
}

async function handleSave() {
  if (!canWrite.value) return
  saving.value = true
  try {
    const payload = {
      top_k: form.value.top_k,
      rerank_enabled: form.value.rerank_enabled,
      rag_general_fallback_mode: form.value.rag_general_fallback_mode,
      rag_general_fallback_model: form.value.rag_general_fallback_model?.trim() || '',
      show_sources: form.value.show_sources,
      site_title: form.value.site_title,
      site_description: form.value.site_description,
      site_logo: form.value.site_logo,
      browser_title: form.value.browser_title,
      site_copyright: form.value.site_copyright,
    }
    await settingsStore.save(payload)
    msg.success('设置已保存')
    // 刷新公开品牌信息，让侧边栏 / 浏览器标题即时更新
    await siteStore.fetchSite()
    form.value = { ...settingsStore.data }
  } catch {
    msg.error('保存失败，请检查配置')
  } finally {
    saving.value = false
  }
}
</script>
