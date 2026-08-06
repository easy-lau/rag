<template>
  <section class="chat-welcome" aria-labelledby="chat-welcome-title">
    <div class="chat-welcome__hero">
      <div class="chat-welcome__mark" aria-hidden="true">
        <n-icon :size="24"><ShieldCheckmarkOutline /></n-icon>
      </div>

      <div class="chat-welcome__copy">
        <p class="chat-welcome__eyebrow">安全知识助手</p>
        <h2 id="chat-welcome-title">{{ greeting }}，今天想查什么？</h2>
        <p>
          从授权知识库中查找可靠答案，系统会同时核对资料范围、适用版本与回答出处。
        </p>
      </div>

      <div class="chat-welcome__trust" aria-label="回答保障">
        <span><i aria-hidden="true"></i>仅检索有权访问的内容</span>
        <span><i aria-hidden="true"></i>回答依据可追溯</span>
      </div>
    </div>

    <div class="chat-welcome__examples" role="group" aria-label="示例问题">
      <p>你可以这样问</p>
      <div class="chat-welcome__example-grid">
        <button
          v-for="(example, index) in examples"
          :key="example"
          type="button"
          class="chat-welcome__example"
          :aria-label="`使用示例问题：${example}`"
          @click="$emit('select-example', example)"
        >
          <span class="chat-welcome__example-index">0{{ index + 1 }}</span>
          <span>{{ example }}</span>
          <n-icon :size="16" aria-hidden="true"><ArrowForwardOutline /></n-icon>
        </button>
      </div>
    </div>

    <!-- 输入与检索配置仍由 ChatInput 统一管理。 -->
    <div class="chat-welcome__composer">
      <slot name="input" />
    </div>
  </section>
</template>

<script setup>
import { computed } from 'vue'
import { NIcon } from 'naive-ui'
import { ArrowForwardOutline, ShieldCheckmarkOutline } from '@vicons/ionicons5'

const props = defineProps({
  siteName: { type: String, default: '知识工作台' },
  userName: { type: String, default: '' },
  examples: {
    type: Array,
    default: () => [
      '公司的报销流程是什么？',
      '如何创建知识库并上传文档？',
      '帮我把这段通知润色得更专业',
    ],
  },
})

defineEmits(['select-example'])

const greeting = computed(() => props.userName ? `你好，${props.userName}` : '你好')
</script>

<style scoped>
.chat-welcome {
  width: min(100%, 900px);
  margin: 0 auto;
}

.chat-welcome__hero {
  position: relative;
  overflow: hidden;
  border: 1px solid color-mix(in srgb, var(--ui-primary) 18%, var(--ui-border));
  border-radius: var(--ui-radius-dialog);
  background:
    linear-gradient(135deg, color-mix(in srgb, var(--ui-primary-subtle) 78%, var(--ui-surface)) 0%, var(--ui-surface) 64%),
    var(--ui-surface);
  padding: clamp(24px, 4vw, 40px);
}

.chat-welcome__hero::after {
  position: absolute;
  right: -76px;
  bottom: -88px;
  width: 250px;
  height: 170px;
  border: 1px solid color-mix(in srgb, var(--ui-primary) 18%, transparent);
  border-radius: 42px;
  background: color-mix(in srgb, var(--ui-primary) 7%, transparent);
  transform: rotate(-12deg);
  content: '';
}

.chat-welcome__mark {
  position: relative;
  z-index: 1;
  display: grid;
  width: 48px;
  height: 48px;
  place-items: center;
  border-radius: 14px;
  background: linear-gradient(180deg, var(--ui-primary) 0%, var(--ui-primary-hover) 100%);
  box-shadow: 0 10px 24px color-mix(in srgb, var(--ui-primary) 22%, transparent);
  color: var(--ui-text-on-primary);
}

.chat-welcome__copy {
  position: relative;
  z-index: 1;
  max-width: 650px;
  margin-top: 24px;
}

.chat-welcome__eyebrow,
.chat-welcome__examples > p {
  margin: 0;
  color: var(--ui-primary);
  font-size: 11px;
  font-weight: 750;
  letter-spacing: .14em;
}

.chat-welcome__copy h2 {
  margin: 9px 0 10px;
  color: var(--ui-text);
  font-size: clamp(26px, 4vw, 38px);
  font-weight: 720;
  line-height: 1.22;
  letter-spacing: -.035em;
}

.chat-welcome__copy > p:last-child {
  margin: 0;
  color: var(--ui-text-secondary);
  font-size: 14px;
  line-height: 1.75;
}

.chat-welcome__trust {
  position: relative;
  z-index: 1;
  display: flex;
  flex-wrap: wrap;
  gap: 9px 20px;
  margin-top: 24px;
  color: var(--ui-text-tertiary);
  font-size: 12px;
}

.chat-welcome__trust span {
  display: inline-flex;
  align-items: center;
  gap: 7px;
}

.chat-welcome__trust i {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--ui-success);
  box-shadow: 0 0 0 3px color-mix(in srgb, var(--ui-success) 12%, transparent);
}

.chat-welcome__examples {
  margin-top: 22px;
}

.chat-welcome__examples > p {
  color: var(--ui-text-tertiary);
  letter-spacing: .08em;
}

.chat-welcome__example-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 10px;
  margin-top: 10px;
}

.chat-welcome__example {
  display: grid;
  min-height: 74px;
  grid-template-columns: auto minmax(0, 1fr) auto;
  align-items: center;
  gap: 9px;
  border: 1px solid var(--ui-border);
  border-radius: var(--ui-radius-card);
  background: var(--ui-surface);
  padding: 13px 14px;
  color: var(--ui-text-secondary);
  cursor: pointer;
  font: inherit;
  font-size: 12px;
  line-height: 1.55;
  text-align: left;
  transition: border-color .18s ease, background .18s ease, color .18s ease, transform .18s ease;
}

.chat-welcome__example:hover {
  border-color: var(--ui-border-focus);
  background: var(--ui-primary-subtle);
  color: var(--ui-primary);
  transform: translateY(-1px);
}

.chat-welcome__example-index {
  color: var(--ui-text-tertiary);
  font-size: 10px;
  font-weight: 750;
  letter-spacing: .05em;
}

.chat-welcome__example :deep(.n-icon) {
  color: var(--ui-icon);
}

.chat-welcome__composer {
  margin-top: 14px;
}

@media (max-width: 767px) {
  .chat-welcome__example-grid { grid-template-columns: 1fr; }
  .chat-welcome__example { min-height: 56px; }
}

@media (max-width: 639px) {
  .chat-welcome__hero { padding: 24px 20px; }
  .chat-welcome__mark { width: 44px; height: 44px; }
  .chat-welcome__copy { margin-top: 19px; }
  .chat-welcome__trust { align-items: flex-start; flex-direction: column; }
}
</style>
