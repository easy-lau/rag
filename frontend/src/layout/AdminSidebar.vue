<template>
  <aside class="w-full h-full flex flex-col bg-[#26384d] text-slate-200">
    <!-- 独立后台品牌区 -->
    <div class="px-5 pt-5 pb-4 border-b border-slate-500/35">
      <router-link v-if="canReturnToChat" to="/chat" class="flex items-center gap-3 group" @click="closeMobileNav">
        <img
          v-if="siteStore.site_logo"
          :src="siteStore.site_logo"
          class="w-10 h-10 rounded-xl object-cover shrink-0 ring-1 ring-white/20"
          alt="站点 Logo"
        />
        <div
          v-else
          class="w-10 h-10 rounded-xl shrink-0 flex items-center justify-center text-white font-bold bg-gradient-to-br from-blue-400 via-indigo-500 to-violet-600 shadow-lg shadow-indigo-950/40"
        >
          {{ (siteStore.site_title || 'R')[0] }}
        </div>
        <div class="min-w-0">
          <div class="text-sm font-semibold text-white truncate group-hover:text-blue-200 transition-colors">
            {{ siteStore.site_title || 'RAG 检索系统' }}
          </div>
          <div class="mt-0.5 flex items-center gap-1.5 text-[11px] text-slate-400">
            <span class="w-1.5 h-1.5 rounded-full bg-emerald-400"></span>
            管理后台
          </div>
        </div>
      </router-link>
      <div v-else class="flex items-center gap-3">
        <img
          v-if="siteStore.site_logo"
          :src="siteStore.site_logo"
          class="w-10 h-10 rounded-xl object-cover shrink-0 ring-1 ring-white/20"
          alt="站点 Logo"
        />
        <div
          v-else
          class="w-10 h-10 rounded-xl shrink-0 flex items-center justify-center text-white font-bold bg-gradient-to-br from-blue-400 via-indigo-500 to-violet-600 shadow-lg shadow-indigo-950/40"
        >
          {{ (siteStore.site_title || 'R')[0] }}
        </div>
        <div class="min-w-0">
          <div class="text-sm font-semibold text-white truncate">{{ siteStore.site_title || 'RAG 检索系统' }}</div>
          <div class="mt-0.5 flex items-center gap-1.5 text-[11px] text-slate-400">
            <span class="w-1.5 h-1.5 rounded-full bg-emerald-400"></span>
            管理后台
          </div>
        </div>
      </div>
    </div>

    <!-- 按业务域分组的后台菜单；具体可见项由 accessibleAdminMenus 统一按权限过滤。 -->
    <nav class="flex-1 overflow-y-auto px-3 py-5 space-y-3">
      <section v-for="group in groupedMenus" :key="group.key">
        <div class="px-3 mb-1.5 text-[11px] font-semibold tracking-[0.1em] text-slate-400">
          {{ group.title }}
        </div>
        <div class="space-y-1">
          <router-link
            v-for="item in group.items"
            :key="item.to"
            :to="item.to"
            class="group flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm transition-all duration-150"
            :class="isActive(item)
              ? 'bg-blue-500 text-white shadow-lg shadow-blue-950/35'
              : 'text-slate-200 hover:bg-white/10 hover:text-white'"
            @click="closeMobileNav"
          >
            <n-icon :size="18" :class="isActive(item) ? 'text-white' : 'text-slate-400 group-hover:text-blue-300'">
              <component :is="item.icon" />
            </n-icon>
            <span class="truncate">{{ item.label }}</span>
            <span
              v-if="isActive(item)"
              class="ml-auto w-1.5 h-1.5 rounded-full bg-white/90 shadow-sm"
              aria-hidden="true"
            ></span>
          </router-link>
        </div>
      </section>

      <div v-if="!groupedMenus.length" class="px-3 py-8 text-center text-xs leading-6 text-slate-500">
        当前账号暂无后台管理权限
      </div>
    </nav>

    <div class="px-5 py-4 border-t border-slate-500/35">
      <router-link
        v-if="canReturnToChat"
        to="/chat"
        class="flex items-center gap-2 text-xs text-slate-400 hover:text-white transition-colors"
        @click="closeMobileNav"
      >
        <n-icon :size="15"><ChatbubbleEllipsesOutline /></n-icon>
        返回智能问答
      </router-link>
    </div>
  </aside>
</template>

<script setup>
import { computed } from 'vue'
import { useRoute } from 'vue-router'
import { NIcon } from 'naive-ui'
import { ChatbubbleEllipsesOutline } from '@vicons/ionicons5'
import { useAuthStore } from '@/stores/auth'
import { useSiteStore } from '@/stores/site'
import { useUiStore } from '@/stores/ui'
import { ADMIN_MENU_GROUPS, accessibleAdminMenus } from '@/router/menus'

const route = useRoute()
const authStore = useAuthStore()
const siteStore = useSiteStore()
const ui = useUiStore()

const adminMenuItems = computed(() => accessibleAdminMenus(authStore))
const canReturnToChat = computed(() => authStore.hasPerm('menu:chat'))

const groupedMenus = computed(() => ADMIN_MENU_GROUPS
  .map(group => ({
    ...group,
    items: adminMenuItems.value.filter(item => item.group === group.key),
  }))
  .filter(group => group.items.length > 0)
)

function isActive(item) {
  return (item.match || [item.to]).some(path => route.path.startsWith(path))
}

function closeMobileNav() {
  ui.mobileNavOpen = false
}
</script>
