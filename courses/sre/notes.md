<!-- markdownlint-disable MD024 MD025 -->

# **Lecture 1 - Intro**

Site Reliability Engineering

**Postmortem** - _blameless_, structured (document) analysis of an incident.

**SPOF (Single Point Of Failure)** - if it fails, it brings down the whole system.

**Rate limiter** - mechanism that controls how many requests are allowed within a period.

## Certificate

**Certificate** - digitally signed document that proves the ownership of a **Public Key Infrastructure (PKI)**.

- SSL/TLS certificates – secure HTTPS websites.

- Code-signing certificates – verify software authenticity.

- Client certificates – authenticate users or devices.

- Root / intermediate certificates – used by Certificate Authorities (CAs) to issue others.

## BGP

**BGP (Border Gateway Protocol)** - routing protocol of the internet.

BGP was designed in the 1980s with _“trust everyone”_ philosophy.

- DNS tells you where someone lives (IP).
- BGP decides how to get there.

---
> Дебаг принтами
