<template>
  <aside class="admin-sidebar w-full h-full flex flex-col">
    <!-- 独立后台品牌区 -->
    <div class="admin-sidebar__brand px-5 pt-5 pb-4">
      <router-link v-if="canReturnToChat" to="/chat" class="flex items-center gap-3 group" @click="closeMobileNav">
        <img
          v-if="siteStore.site_logo"
          :src="siteStore.site_logo"
          class="admin-sidebar__logo w-10 h-10 object-cover shrink-0"
          alt="站点 Logo"
        />
        <div
          v-else
          class="admin-sidebar__logo admin-sidebar__logo--fallback w-10 h-10 shrink-0 flex items-center justify-center font-bold"
        >
          {{ (siteStore.site_title || 'R')[0] }}
        </div>
        <div class="min-w-0">
          <div class="admin-sidebar__brand-title text-sm font-semibold truncate transition-colors">
            {{ siteStore.site_title || 'RAG 检索系统' }}
          </div>
          <div class="admin-sidebar__brand-caption mt-0.5 flex items-center gap-1.5 text-[11px]">
            <span class="admin-sidebar__brand-status w-1.5 h-1.5 rounded-full"></span>
            管理后台
          </div>
        </div>
      </router-link>
      <div v-else class="flex items-center gap-3">
        <img
          v-if="siteStore.site_logo"
          :src="siteStore.site_logo"
          class="admin-sidebar__logo w-10 h-10 object-cover shrink-0"
          alt="站点 Logo"
        />
        <div
          v-else
          class="admin-sidebar__logo admin-sidebar__logo--fallback w-10 h-10 shrink-0 flex items-center justify-center font-bold"
        >
          {{ (siteStore.site_title || 'R')[0] }}
        </div>
        <div class="min-w-0">
          <div class="admin-sidebar__brand-title text-sm font-semibold truncate">{{ siteStore.site_title || 'RAG 检索系统' }}</div>
          <div class="admin-sidebar__brand-caption mt-0.5 flex items-center gap-1.5 text-[11px]">
            <span class="admin-sidebar__brand-status w-1.5 h-1.5 rounded-full"></span>
            管理后台
          </div>
        </div>
      </div>
    </div>

    <!-- 按业务域分组的后台菜单；具体可见项由 accessibleAdminMenus 统一按权限过滤。 -->
    <nav class="flex flex-1 flex-col gap-4 overflow-y-auto px-3 py-5">
      <section v-for="group in groupedMenus" :key="group.key" class="admin-sidebar__group">
        <div class="admin-sidebar__group-title px-3 mb-1.5 text-[11px] font-semibold tracking-[0.1em]">
          {{ group.label }}
        </div>
        <div class="space-y-1.5">
          <router-link
            v-for="item in group.items"
            :key="item.to"
            :to="item.to"
            class="admin-sidebar__item group flex items-center gap-3 px-3 py-2.5 text-sm transition-all duration-150"
            :class="isActive(item)
              ? 'admin-sidebar__item--active'
              : 'admin-sidebar__item--idle'"
            :aria-current="isActive(item) ? 'page' : undefined"
            @click="closeMobileNav"
          >
            <n-icon :size="18" class="admin-sidebar__item-icon">
              <component :is="item.icon" />
            </n-icon>
            <span class="truncate">{{ item.label }}</span>
          </router-link>
        </div>
      </section>

      <div v-if="!groupedMenus.length" class="admin-sidebar__empty px-3 py-8 text-center text-xs leading-6">
        当前账号暂无后台管理权限
      </div>
    </nav>

  </aside>
</template>

<script setup>
import { computed } from 'vue'
import { useRoute } from 'vue-router'
import { NIcon } from 'naive-ui'
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

<style scoped>
.admin-sidebar {
  color: var(--admin-sidebar-text);
  background: var(--admin-sidebar-bg);
}

.admin-sidebar__brand {
  border-bottom: 1px solid var(--admin-sidebar-divider);
}

.admin-sidebar__logo {
  overflow: hidden;
  border: 1px solid var(--admin-sidebar-logo-border);
  border-radius: var(--admin-ui-radius-control);
}

.admin-sidebar__logo--fallback {
  background: var(--admin-sidebar-logo-bg);
  color: var(--admin-sidebar-logo-text);
}

.admin-sidebar__brand-title { color: var(--admin-sidebar-text-strong); }
.group:hover .admin-sidebar__brand-title { color: var(--admin-sidebar-accent); }
.admin-sidebar__brand-caption { color: var(--admin-sidebar-muted); }
.admin-sidebar__brand-status {
  background: var(--admin-sidebar-accent);
  box-shadow: 0 0 0 3px var(--admin-sidebar-active);
}

/* 业务域之间用分隔线和留白区分，避免「智能路由」与系统管理项视觉上连成一组。 */
.admin-sidebar__group + .admin-sidebar__group {
  padding-top: 14px;
  border-top: 1px solid var(--admin-sidebar-divider);
}

.admin-sidebar__group-title {
  color: var(--admin-sidebar-muted);
}

.admin-sidebar__item {
  position: relative;
  min-height: 42px;
  color: var(--admin-sidebar-text);
  border: 1px solid transparent;
  border-radius: var(--admin-ui-radius-control);
}

.admin-sidebar__item--idle:hover {
  color: var(--admin-sidebar-accent);
  background: var(--admin-sidebar-hover);
}

.admin-sidebar__item--active {
  color: var(--admin-sidebar-accent);
  background: var(--admin-sidebar-active);
  border-color: color-mix(in srgb, var(--admin-sidebar-accent) 12%, var(--admin-sidebar-divider));
  font-weight: 650;
}

.admin-sidebar__item--active::before {
  content: '';
  position: absolute;
  top: 10px;
  bottom: 10px;
  left: 0;
  width: 3px;
  border-radius: var(--admin-ui-radius-pill);
  background: var(--admin-sidebar-accent);
}

.admin-sidebar__item-icon { color: var(--admin-sidebar-muted); }
.admin-sidebar__item--idle:hover .admin-sidebar__item-icon,
.admin-sidebar__item--active .admin-sidebar__item-icon { color: var(--admin-sidebar-accent); }
.admin-sidebar__empty { color: var(--admin-sidebar-muted); }
</style>
